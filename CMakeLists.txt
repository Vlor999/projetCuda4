cmake_minimum_required(VERSION 3.16)
project(ass04-3dgs-full LANGUAGES CXX CUDA VERSION 1.1)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CC "89" CACHE STRING "Which compute capability to build for? e.g. 75, 86, 89, ...")

set(DEPENDENCIES_DIR "${CMAKE_SOURCE_DIR}/dependencies")
add_subdirectory(dependencies/glm)

####################################################################
# Helper Functions
add_library(render_common
	src/dataset.cpp
	include/dataset.h
)
target_include_directories(render_common PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper" "${DEPENDENCIES_DIR}")
set_target_properties(render_common PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(render_common PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(render_common PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(render_common PRIVATE -Wall -Wextra $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

####################################################################
# GPU Reference Implementation
add_library(render_gpu_ref
	src/renderer_reference.cu
)
target_include_directories(render_gpu_ref PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper" "${DEPENDENCIES_DIR}")
set_target_properties(render_gpu_ref PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(render_gpu_ref PROPERTIES CXX_STANDARD 17)
target_compile_options(render_gpu_ref PRIVATE -Xcudafe="--diag_suppress=20012" -Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v -lineinfo $<$<CONFIG:DEBUG>:-G>)


####################################################################
# GPU Implementation
add_library(render_gpu
	src/renderer.cu
)
target_include_directories(render_gpu PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper" "${DEPENDENCIES_DIR}")
set_target_properties(render_gpu PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(render_gpu PROPERTIES CXX_STANDARD 17)
target_compile_options(render_gpu PRIVATE -Xcudafe="--diag_suppress=20012" -Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v -lineinfo $<$<CONFIG:DEBUG>:-G>)


####################################################################
# Executable
add_executable(3dgs-full
	src/main.cpp
)
target_include_directories(3dgs-full PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper" "${DEPENDENCIES_DIR}")
set_target_properties(3dgs-full PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(3dgs-full PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(3dgs-full PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(3dgs-full PRIVATE -Wall -Wextra -Wno-missing-field-initializers $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

# Link all together
target_link_libraries(3dgs-full render_common render_gpu_ref render_gpu glm)
