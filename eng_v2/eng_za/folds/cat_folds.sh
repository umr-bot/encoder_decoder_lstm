# combine parts to make train val and test for folds
cat part{1,2,3} > "fold1/train"
cat part4 > "fold1/val"
cat part5 > "fold1/test"

cat part{2,3,4} > "fold2/train"
cat part5 > "fold2/val"
cat part1 > "fold2/test"

cat part{3,4,5} > "fold3/train"
cat part1 > "fold3/val"
cat part2 > "fold3/test"

cat part{4,5,1} > "fold4/train"
cat part2 > "fold4/val"
cat part3 > "fold4/test"

cat part{5,1,2} > "fold5/train"
cat part3 > "fold5/val"
cat part4 > "fold5/test"

