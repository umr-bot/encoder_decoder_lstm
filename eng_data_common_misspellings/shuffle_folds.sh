# shuffle folds
#dd if=/dev/random of=myrand count=1024 # to generate myrand file
root_pwd=$(pwd)
prefix="unshuffled_folds"
for fold in ${prefix}/*
do
    echo "Fold ${fold} being processed"
    for fn in ${fold}/*
    do
        echo "Processing file ${fn}"
        #cat ${fn} | wc -l
        string=${fn}
        stripped_string=${string#"$prefix"}
        shuf_out_fn="folds${stripped_string}"
        shuf --random-source=myrand ${fn} > ${shuf_out_fn} 
    done
done
