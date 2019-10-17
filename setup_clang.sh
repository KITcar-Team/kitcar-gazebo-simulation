# this script sets up environment variables to use the clang compiler

if [ -z "$1" ]; then
 VERSION=4.0
else
 VERSION=$1
fi

export CC=/usr/bin/clang-$VERSION
export CXX=/usr/bin/clang++-$VERSION

export C_INCLUDE_PATH=$(llvm-config-$VERSION --includedir)
export CPLUS_INCLUDE_PATH=$(llvm-config-$VERSION --includedir)
export LIBRARY_PATH=$(llvm-config-$VERSION --libdir)
