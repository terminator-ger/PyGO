#! /bin/bash
if [!-d ./thirdparty]; then
    mkdir -p thirdparty;
fi;

cd thirdparty
git clone git@github.com:terminator-ger/python-lsd.git
git clone git@github.com:terminator-ger/XiaohuLuVPDetection.git
