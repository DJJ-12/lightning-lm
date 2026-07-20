#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace lightning::runtime {

enum class QueuePopResult {
    MESSAGE,
    TIMEOUT,
    CLOSED
};

// 简单的线程安全 FIFO。Open() 开始接收，Close() 停止接收。
// Close(true) 会处理完已经入队的数据；Close(false) 会立即清空队列。
template <typename T>
class MessageQueue {
   public:
    void Open() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        open_ = true;
    }

    bool Push(T value, std::size_t* depth = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_) {
            return false;
        }
        queue_.push_back(std::move(value));
        if (depth) {
            *depth = queue_.size();
        }
        cv_.notify_one();
        return true;
    }

    QueuePopResult WaitPop(T* value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty() || !open_; });
        return PopLocked(value);
    }

    template <typename Rep, typename Period>
    QueuePopResult WaitPopFor(T* value, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this]() { return !queue_.empty() || !open_; })) {
            return QueuePopResult::TIMEOUT;
        }
        return PopLocked(value);
    }

    std::size_t Close(bool drain) {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = false;
        const std::size_t pending = queue_.size();
        if (!drain) {
            queue_.clear();
        }
        cv_.notify_all();
        return pending;
    }

    std::size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

   private:
    QueuePopResult PopLocked(T* value) {
        if (!queue_.empty()) {
            *value = std::move(queue_.front());
            queue_.pop_front();
            return QueuePopResult::MESSAGE;
        }
        return QueuePopResult::CLOSED;
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> queue_;
    bool open_ = false;
};

}  // namespace lightning::runtime
