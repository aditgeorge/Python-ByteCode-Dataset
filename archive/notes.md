python -m py_compile s007069827.py
python pyc_to_json.py ./temp/s001696467.cpython-310.pyc -o my_dataset.json
python json_to_pyc.py my_dataset.json -o my_new_file.pyc
