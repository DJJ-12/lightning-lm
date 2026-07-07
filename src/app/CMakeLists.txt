add_executable(run_lightning
        run_lightning.cc
)

target_link_libraries(run_lightning
        ${PROJECT_NAME}.libs
        ${third_party_libs}
)

install(TARGETS run_lightning
        DESTINATION lib/${PROJECT_NAME})
