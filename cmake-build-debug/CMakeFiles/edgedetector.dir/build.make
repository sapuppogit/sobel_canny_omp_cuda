# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/83/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/83/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/saem/programs/CLionProjects/edgedetector

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/saem/programs/CLionProjects/edgedetector/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/edgedetector.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/edgedetector.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/edgedetector.dir/flags.make

CMakeFiles/edgedetector.dir/main.cpp.o: CMakeFiles/edgedetector.dir/flags.make
CMakeFiles/edgedetector.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/edgedetector.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/edgedetector.dir/main.cpp.o -c /home/saem/programs/CLionProjects/edgedetector/main.cpp

CMakeFiles/edgedetector.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgedetector.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saem/programs/CLionProjects/edgedetector/main.cpp > CMakeFiles/edgedetector.dir/main.cpp.i

CMakeFiles/edgedetector.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgedetector.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saem/programs/CLionProjects/edgedetector/main.cpp -o CMakeFiles/edgedetector.dir/main.cpp.s

CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o: CMakeFiles/edgedetector.dir/flags.make
CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o: ../Sobel/Sobel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o -c /home/saem/programs/CLionProjects/edgedetector/Sobel/Sobel.cpp

CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saem/programs/CLionProjects/edgedetector/Sobel/Sobel.cpp > CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.i

CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saem/programs/CLionProjects/edgedetector/Sobel/Sobel.cpp -o CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.s

CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o: CMakeFiles/edgedetector.dir/flags.make
CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o: ../Sobel/ompSobel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o -c /home/saem/programs/CLionProjects/edgedetector/Sobel/ompSobel.cpp

CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saem/programs/CLionProjects/edgedetector/Sobel/ompSobel.cpp > CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.i

CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saem/programs/CLionProjects/edgedetector/Sobel/ompSobel.cpp -o CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.s

CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o: CMakeFiles/edgedetector.dir/flags.make
CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o: ../Canny/Canny.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o -c /home/saem/programs/CLionProjects/edgedetector/Canny/Canny.cpp

CMakeFiles/edgedetector.dir/Canny/Canny.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgedetector.dir/Canny/Canny.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saem/programs/CLionProjects/edgedetector/Canny/Canny.cpp > CMakeFiles/edgedetector.dir/Canny/Canny.cpp.i

CMakeFiles/edgedetector.dir/Canny/Canny.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgedetector.dir/Canny/Canny.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saem/programs/CLionProjects/edgedetector/Canny/Canny.cpp -o CMakeFiles/edgedetector.dir/Canny/Canny.cpp.s

CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o: CMakeFiles/edgedetector.dir/flags.make
CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o: ../Canny/ompCanny.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o -c /home/saem/programs/CLionProjects/edgedetector/Canny/ompCanny.cpp

CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saem/programs/CLionProjects/edgedetector/Canny/ompCanny.cpp > CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.i

CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saem/programs/CLionProjects/edgedetector/Canny/ompCanny.cpp -o CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.s

CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o: CMakeFiles/edgedetector.dir/flags.make
CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o: ../Sobel/cudaSobel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/saem/programs/CLionProjects/edgedetector/Sobel/cudaSobel.cu -o CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o

CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o: CMakeFiles/edgedetector.dir/flags.make
CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o: ../Canny/cudaCanny.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/saem/programs/CLionProjects/edgedetector/Canny/cudaCanny.cu -o CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o

CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target edgedetector
edgedetector_OBJECTS = \
"CMakeFiles/edgedetector.dir/main.cpp.o" \
"CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o" \
"CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o" \
"CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o" \
"CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o" \
"CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o" \
"CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o"

# External object files for target edgedetector
edgedetector_EXTERNAL_OBJECTS =

CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/main.cpp.o
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/build.make
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudastereo.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_stitching.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_superres.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_videostab.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_aruco.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_bgsegm.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_bioinspired.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_ccalib.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cvv.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_dpm.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_face.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_freetype.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_fuzzy.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_hfs.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_img_hash.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_line_descriptor.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_optflow.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_reg.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_rgbd.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_saliency.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_stereo.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_structured_light.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_surface_matching.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_tracking.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_ximgproc.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_xobjdetect.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_xphoto.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_shape.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudawarping.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_photo.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudafilters.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_datasets.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_plot.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_text.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_dnn.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_ml.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_video.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_calib3d.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_features2d.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_highgui.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_videoio.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_flann.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_objdetect.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_imgproc.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_core.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: /usr/local/lib/libopencv_cudev.so.3.4.1
CMakeFiles/edgedetector.dir/cmake_device_link.o: CMakeFiles/edgedetector.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CUDA device code CMakeFiles/edgedetector.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/edgedetector.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/edgedetector.dir/build: CMakeFiles/edgedetector.dir/cmake_device_link.o

.PHONY : CMakeFiles/edgedetector.dir/build

# Object files for target edgedetector
edgedetector_OBJECTS = \
"CMakeFiles/edgedetector.dir/main.cpp.o" \
"CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o" \
"CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o" \
"CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o" \
"CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o" \
"CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o" \
"CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o"

# External object files for target edgedetector
edgedetector_EXTERNAL_OBJECTS =

edgedetector: CMakeFiles/edgedetector.dir/main.cpp.o
edgedetector: CMakeFiles/edgedetector.dir/Sobel/Sobel.cpp.o
edgedetector: CMakeFiles/edgedetector.dir/Sobel/ompSobel.cpp.o
edgedetector: CMakeFiles/edgedetector.dir/Canny/Canny.cpp.o
edgedetector: CMakeFiles/edgedetector.dir/Canny/ompCanny.cpp.o
edgedetector: CMakeFiles/edgedetector.dir/Sobel/cudaSobel.cu.o
edgedetector: CMakeFiles/edgedetector.dir/Canny/cudaCanny.cu.o
edgedetector: CMakeFiles/edgedetector.dir/build.make
edgedetector: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudastereo.so.3.4.1
edgedetector: /usr/local/lib/libopencv_stitching.so.3.4.1
edgedetector: /usr/local/lib/libopencv_superres.so.3.4.1
edgedetector: /usr/local/lib/libopencv_videostab.so.3.4.1
edgedetector: /usr/local/lib/libopencv_aruco.so.3.4.1
edgedetector: /usr/local/lib/libopencv_bgsegm.so.3.4.1
edgedetector: /usr/local/lib/libopencv_bioinspired.so.3.4.1
edgedetector: /usr/local/lib/libopencv_ccalib.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cvv.so.3.4.1
edgedetector: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
edgedetector: /usr/local/lib/libopencv_dpm.so.3.4.1
edgedetector: /usr/local/lib/libopencv_face.so.3.4.1
edgedetector: /usr/local/lib/libopencv_freetype.so.3.4.1
edgedetector: /usr/local/lib/libopencv_fuzzy.so.3.4.1
edgedetector: /usr/local/lib/libopencv_hfs.so.3.4.1
edgedetector: /usr/local/lib/libopencv_img_hash.so.3.4.1
edgedetector: /usr/local/lib/libopencv_line_descriptor.so.3.4.1
edgedetector: /usr/local/lib/libopencv_optflow.so.3.4.1
edgedetector: /usr/local/lib/libopencv_reg.so.3.4.1
edgedetector: /usr/local/lib/libopencv_rgbd.so.3.4.1
edgedetector: /usr/local/lib/libopencv_saliency.so.3.4.1
edgedetector: /usr/local/lib/libopencv_stereo.so.3.4.1
edgedetector: /usr/local/lib/libopencv_structured_light.so.3.4.1
edgedetector: /usr/local/lib/libopencv_surface_matching.so.3.4.1
edgedetector: /usr/local/lib/libopencv_tracking.so.3.4.1
edgedetector: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
edgedetector: /usr/local/lib/libopencv_ximgproc.so.3.4.1
edgedetector: /usr/local/lib/libopencv_xobjdetect.so.3.4.1
edgedetector: /usr/local/lib/libopencv_xphoto.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
edgedetector: /usr/local/lib/libopencv_shape.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudawarping.so.3.4.1
edgedetector: /usr/local/lib/libopencv_photo.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudafilters.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
edgedetector: /usr/local/lib/libopencv_datasets.so.3.4.1
edgedetector: /usr/local/lib/libopencv_plot.so.3.4.1
edgedetector: /usr/local/lib/libopencv_text.so.3.4.1
edgedetector: /usr/local/lib/libopencv_dnn.so.3.4.1
edgedetector: /usr/local/lib/libopencv_ml.so.3.4.1
edgedetector: /usr/local/lib/libopencv_video.so.3.4.1
edgedetector: /usr/local/lib/libopencv_calib3d.so.3.4.1
edgedetector: /usr/local/lib/libopencv_features2d.so.3.4.1
edgedetector: /usr/local/lib/libopencv_highgui.so.3.4.1
edgedetector: /usr/local/lib/libopencv_videoio.so.3.4.1
edgedetector: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.1
edgedetector: /usr/local/lib/libopencv_flann.so.3.4.1
edgedetector: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
edgedetector: /usr/local/lib/libopencv_objdetect.so.3.4.1
edgedetector: /usr/local/lib/libopencv_imgproc.so.3.4.1
edgedetector: /usr/local/lib/libopencv_core.so.3.4.1
edgedetector: /usr/local/lib/libopencv_cudev.so.3.4.1
edgedetector: CMakeFiles/edgedetector.dir/cmake_device_link.o
edgedetector: CMakeFiles/edgedetector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable edgedetector"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/edgedetector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/edgedetector.dir/build: edgedetector

.PHONY : CMakeFiles/edgedetector.dir/build

CMakeFiles/edgedetector.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/edgedetector.dir/cmake_clean.cmake
.PHONY : CMakeFiles/edgedetector.dir/clean

CMakeFiles/edgedetector.dir/depend:
	cd /home/saem/programs/CLionProjects/edgedetector/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/saem/programs/CLionProjects/edgedetector /home/saem/programs/CLionProjects/edgedetector /home/saem/programs/CLionProjects/edgedetector/cmake-build-debug /home/saem/programs/CLionProjects/edgedetector/cmake-build-debug /home/saem/programs/CLionProjects/edgedetector/cmake-build-debug/CMakeFiles/edgedetector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/edgedetector.dir/depend
