cd fairgraph/dataset
python graphsaint/setup.py build_ext --inplace
find . -name "*.so" -exec mv {} graphsaint/ \;
rm -rf build