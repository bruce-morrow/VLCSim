for TEST_FILE in $(locate "**/VLCSim/VLCSim/**/tests/test_*.py")
do
  echo "running ${TEST_FILE##*/}"
  python3 -m unittest "$TEST_FILE"
  echo -e "\n\n"
done
