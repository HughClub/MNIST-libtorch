# MNIST with libtorch

A simple repo for beginner to train on MNIST with libtorch(cpu version)

## libtorch

1. Download libtorch from https://pytorch.org/get-started/locally/ or `wget` the [latest nightly build](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip).
2. `unzip` your libtorch and set the environment variable `YOUR_LIBTORCH_FOLDER` to the path of the unzipped folder.

## editor hint

> For VSCode

You should set the `includePath` for VSCode to search the include headers.

```bash
YOUR_LIBTORCH_FOLDER="" # fill it
LIBTORCH_INCLUDE="${YOUR_LIBTORCH_FOLDER}/include"
DEEPER_LIBTORCH="${LIBTORCH_INCLUDE}/torch/csrc/api/include"
```

Create `.vscode/c_cpp_properties.json` and set includePath. You can take the following as an example

```json
{
    "configurations": [
        {
            "name": "libtorch",
            "includePath": [
                "./libtorch/include",
                "./libtorch/include/torch/csrc/api/include/"
            ]
        }
    ],
    "version": 4
}
```

## training

There are three relative files.
- [main.cpp](main.cpp): show simple standard training pipeline here
- [net.hpp](net.hpp): the file where the real model hides
- [utils.hpp](utils.hpp): some utils

For simplicity, I don't divide the headers into .h in the include and .cpp in the src and compile they into .a or .so files.

You can build and run your models like:

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$YOUR_LIBTORCH_FOLDER ..
make
cd ..
./build/mnist # It depends on what you specify your data path in your code
```

## result

Without specific seed, cmd:`./build/mnist 512 1e-3 128 150` could achive a test accuracy of 0.8965 with a model size of 216kB (MLP).

Without specific seed, cmd:`./build/mnist 512 1e-4 32 150` could achive a test accuracy of 0.9267 with a model size of 58kB (Conv).