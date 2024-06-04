cd fairgraph/dataset
python3 graphsaint/setup.py build_ext --inplace
find . -name "*.so" -exec mv {} graphsaint/ \;
rm -rf build