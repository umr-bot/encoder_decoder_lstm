# shuffle folds
#dd if=/dev/random of=myrand count=1024 # to generate myrand file
root_pwd=$(pwd)
prefix="folds_enc_dec"
for fold in folds_enc_dec/*
do
    echo "Fold ${fold} being processed"
    for fn in ${fold}/*
    do
        echo "Processing file ${fn}"
        string=${fn}
        stripped_string=${string#"$prefix"}
        shuf_out_fn="shuffled_enc_dec_folds${stripped_string}"
        shuf --random-source=myrand ${fn} > ${shuf_out_fn} 
    done
done
