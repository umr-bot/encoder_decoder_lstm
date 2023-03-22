# combine parts to make train val and test for folds
cat fold{1,2,3} > "foldset1/train"
cat fold4 > "foldset1/val"
cat fold5 > "foldset1/test"

cat fold{2,3,4} > "foldset2/train"
cat fold5 > "foldset2/val"
cat fold1 > "foldset2/test"

cat fold{3,4,5} > "foldset3/train"
cat fold1 > "foldset3/val"
cat fold2 > "foldset3/test"

cat fold{4,5,1} > "foldset4/train"
cat fold2 > "foldset4/val"
cat fold3 > "foldset4/test"

cat fold{5,1,2} > "foldset5/train"
cat fold3 > "foldset5/val"
cat fold4 > "foldset5/test"

