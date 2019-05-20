# Synopsis

Lightweight `c++` wrapper for `PETSc` aiming to enable a more
epressive and simpler usage of `PETSc` with minimal overhead.

```c++
int main(int main(int argc, char *argv[]){

  // Initialize Petsc
  Petscpp::App app(argc, argv);

  int const N = 1600;

  /* Wrapper for PETSc::Vec struct. Initialize a parallel vector of
   * size N, with default value of 3.1
   */
  Petscpp::Vector v1(N, 3.1);

  /* Scale `v1` and assign to a new vector with same layout as v1,
   * with contents 3.1*2
   */
  Petscpp::Vector v2 = 2*v1;

  /* Add `v1` and `v2`, and write to a new vector with same parallel
   * layout as `v1` and `v2`
   */
  Petscpp::Vector v3 = v1 + v2;

  // Create a new vector of dimension 2 for demo purposes
  Petscpp::Vector v4(2);

  /* (`v1`, `v2` , `v3`) have compatible parallel layouts. Following
   * line performs lazy evaluation: original `v4` (different layout as
   * `v1`, `v2`, `v3` is destroyed, and a new vector with compatible
   * layout to (`v1`, `v2`, `v3`) is created. No temporaries are created.
   */
  v4 = v1 + v2 + v3;

  // No need to manually clean up PETSc objects

  return 0;
}
```

**NOTE: Treat this repo as incomplete. This was originally started in
~2015 and has not been completed or updated since.**

# Requirements

- PETSc (used 3.4.4; expect breaking changes in newer versions)
- MPICH
- ATLAS (recommended)
- Eigen
- c++11+ compiler

# Usage

See the [examples](examples) directory.

# Build

1. [Build](#building-petsc) Petsc
2. Build
   ```bash
   cd path-to-petscpp-root
   mkdir build
   cd build
   cmake ..
   make
   ```

# Building PETSc

(Built with `petsc-3.4.4`). Note: there are or were often breaking
changes between different PETSc releases)

Consider building `PETSc` with an optimized linear algebra system
(see example below). An optimized LA system can lead to a
considerable performance boost.

1. Install MPICH
  * Install =MPICH=
     ```bash
     yum install mpich2
     ```
  * Load the MPI module
     ```bash
     module load mpi/mpich-x86_64
     ```
     You can add the above line to `~/.bash_profile` so that MPI
     module is loaded on login
2. Get ATLAS
     1. Download `ATLAS` (tested with `atlas-3.10.3`)
     2. Build ATLAS
        - If using INTEL CPU, boot the OS with `intel_pstate`
          disabled. This can be achieved for example by appending
          `intel_pstate` to boot options.
        - Disable CPU frequency scaling
        ```bash
        echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governo
        echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governo
        # ...
        ```
    3. Configure `ATLAS`. Example build process:
    ```bash
     cd /path/to/atlas/dir
     mkdir name_of_build_directory && cd name_of_build_directory
     ../configure --with-netlib-lapack-tarfile=/path/to/lapak/tarbal.tgz \
         -Fa alg \
         -fPIC
     make build
     make check
     make ptcheck
     make time
    ```
3. Build PETSc
  1. Configure and build `PETSc` (tested with `petsc-3.4.4`). Set
     appropriate flags
  ```bash
  ./configure --with-c-support=1 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 \
              --with-pic \
              --with-debugging=0
              --with-blas-lapack-dir=/path/to/atlast/build/dir/lib \
              --with-x=0 \
              --with-superlu=1
              --download-superlu \
              --with-superlu_dist=1 \
              --download-superlu_dist \
              --with-parmetis=1 \
              --download-parmetis \
              --with-metis=1 \
              --download-metis \
              --with-dynamic-loading=0 \
              --with-shared-libraries=1 \
              --download-mpich
  make PETSC_DIR=/opt/petsc-3.4.4 PETSC_ARCH=arch-linux2-c-opt all
  ```
  2. Set needed PETSc environmental variables by adding following to
     `~/.bashrc`
  ```bash
  $ export PETSC_DIR=/path/to/petsc/dir
  $ export PETSC_ARCH=/path/to/petsc/arch/build
  ```
  3. Install `cmake`
  4. Download Eigen
  5. Add path to Eigen as an env variable in `~/.bashrc`
  ```bash
  # ...
  export EIGEN_DIR=/path/to/eigen/root/dir
  ```
  6. Build
  ```bash
  cd /path/to/petscpp/root/dir/
  mkdir build && cd build
  cmake ..
  make
  ```
  7. Run PETSc tests
  ```bash
  cd /path/to/executable
  mpiexec -n <number-of-procs> ./name_of_executable
  ```

  Limit the processor number up to the number of available physical
  cores when using MPI.
