echoerr() { echo "$@" 1>&2; }

echo "dataset, model_type, algorithm, D1_size, D2_size, D1_zeros, D2_zeros, D1_pred_zeros, D2_pred_zeros, M1_acc, M1_test_acc, M2_acc, M2_test_acc, M1_acc_on_D2, M1_acc_on_D2_test, M2_acc_on_D1, M2_acc_on_D1_test, D1_rec, D1_norec, cf1_inv_by_M2, D2_rec, D2_norec, cf2_inv_by_M1, M1_crossacc, M2_crossacc, comment" > ../results/all_tables.csv

echoerr "################ temporal - LR - AR"
python3 main.py --dataset=temporal --classifier=lr --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - LR - CF" ###RF -> LR
python3 main.py --dataset=temporal --classifier=lr --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - LR - Causal [SKIP]" ###RF -> LR ###[SKIP]
python3 main.py --dataset=temporal --classifier=lr --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ temporal - RF - AR"
python3 main.py --dataset=temporal --classifier=rf --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - RF - CF"
python3 main.py --dataset=temporal --classifier=rf --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - RF - Causal [SKIP]" ###[SKIP]
python3 main.py --dataset=temporal --classifier=rf --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ temporal - XGB - AR"
python3 main.py --dataset=temporal --classifier=xgb --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - XGB - CF"
python3 main.py --dataset=temporal --classifier=xgb --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - XGB - Causal [SKIP]" ###[SKIP]
python3 main.py --dataset=temporal --classifier=xgb --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ temporal - SVM - AR"
python3 main.py --dataset=temporal --classifier=svm --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - SVM - CF"
python3 main.py --dataset=temporal --classifier=svm --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ temporal - SVM - Causal [SKIP]" ###[SKIP]
#python3 main.py --dataset=temporal --classifier=svm --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ temporal - NN1 - AR"
python3 main.py --dataset=temporal --classifier=nn --method=AR --comment=COM --arch 10 10 5 >> ../results/all_tables.csv
echoerr "################ temporal - NN1 - CF"
python3 main.py --dataset=temporal --classifier=nn --method=CF --comment=COM --arch 10 10 5 >> ../results/all_tables.csv
echoerr "################ temporal - NN1 - Causal [SKIP]" ###[SKIP]
python3 main.py --dataset=temporal --classifier=nn --method=c12 --comment=COM --arch 10 10 5 >> ../results/all_tables.csv

echoerr "################ temporal - NN2 - AR" ###NN1 -> NN2
python3 main.py --dataset=temporal --classifier=nn --method=AR --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv
echoerr "################ temporal - NN2 - CF" ###NN1 -> NN2
python3 main.py --dataset=temporal --classifier=nn --method=CF --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv
echoerr "################ temporal - NN2 - Causal [SKIP]" ###NN1 -> NN2 ###[SKIP]
python3 main.py --dataset=temporal --classifier=nn --method=c12 --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv



echoerr "################ geospatial - LR - AR"
python3 main.py --dataset=geospatial --classifier=lr --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - LR - CF" ###RF -> LR
python3 main.py --dataset=geospatial --classifier=lr --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - LR - Causal [SKIP]" ###RF -> LR ###[SKIP]
python3 main.py --dataset=geospatial --classifier=lr --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ geospatial - RF - AR"
python3 main.py --dataset=geospatial --classifier=rf --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - RF - CF"
python3 main.py --dataset=geospatial --classifier=rf --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - RF - Causal [SKIP]" ###[SKIP]
python3 main.py --dataset=geospatial --classifier=rf --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ geospatial - XGB - AR"
python3 main.py --dataset=geospatial --classifier=xgb --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - XGB - CF"
python3 main.py --dataset=geospatial --classifier=xgb --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - XGB - Causal [SKIP]" ###[SKIP]
python3 main.py --dataset=geospatial --classifier=xgb --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ geospatial - SVM - AR"
python3 main.py --dataset=geospatial --classifier=svm --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - SVM - CF"
python3 main.py --dataset=geospatial --classifier=svm --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ geospatial - SVM - Causal [SKIP]"
#python3 main.py --dataset=geospatial --classifier=svm --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ geospatial - NN1 - AR"
python3 main.py --dataset=geospatial --classifier=nn --method=AR --comment=COM --arch 10 10 5 >> ../results/all_tables.csv
echoerr "################ geospatial - NN1 - CF"
python3 main.py --dataset=geospatial --classifier=nn --method=CF --comment=COM --arch 10 10 5 >> ../results/all_tables.csv
echoerr "################ geospatial - NN1 - Causal [SKIP]" ###[SKIP]
python3 main.py --dataset=geospatial --classifier=nn --method=c12 --comment=COM --arch 10 10 5 >> ../results/all_tables.csv

echoerr "################ geospatial - NN2 - AR" ###NN1 -> NN2
python3 main.py --dataset=geospatial --classifier=nn --method=AR --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv
echoerr "################ geospatial - NN2 - CF" ###NN1 -> NN2
python3 main.py --dataset=geospatial --classifier=nn --method=CF --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv
echoerr "################ geospatial - NN2 - Causal [SKIP]" ###NN1 -> NN2 ###[SKIP]
python3 main.py --dataset=geospatial --classifier=nn --method=c12 --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv



echoerr "################ correction - LR - AR"
python3 main.py --dataset=correction --classifier=lr --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - LR - CF" ###RF -> LR
python3 main.py --dataset=correction --classifier=lr --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - LR - Causal" ###RF -> LR
python3 main.py --dataset=correction --classifier=lr --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ correction - RF - AR"
python3 main.py --dataset=correction --classifier=rf --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - RF - CF"
python3 main.py --dataset=correction --classifier=rf --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - RF - Causal"
python3 main.py --dataset=correction --classifier=rf --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ correction - XGB - AR"
python3 main.py --dataset=correction --classifier=xgb --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - XGB - CF"
python3 main.py --dataset=correction --classifier=xgb --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - XGB - Causal"
python3 main.py --dataset=correction --classifier=xgb --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ correction - SVM - AR"
python3 main.py --dataset=correction --classifier=svm --method=AR --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - SVM - CF"
python3 main.py --dataset=correction --classifier=svm --method=CF --comment=COM >> ../results/all_tables.csv
echoerr "################ correction - SVM - Causal [SKIP]"
#python3 main.py --dataset=correction --classifier=svm --method=c12 --comment=COM >> ../results/all_tables.csv

echoerr "################ correction - NN1 - AR"
python3 main.py --dataset=correction --classifier=nn --method=AR --comment=COM --arch 10 10 5 >> ../results/all_tables.csv
echoerr "################ correction - NN1 - CF"
python3 main.py --dataset=correction --classifier=nn --method=CF --comment=COM --arch 10 10 5 >> ../results/all_tables.csv
echoerr "################ correction - NN1 - Causal"
python3 main.py --dataset=correction --classifier=nn --method=c12 --comment=COM --arch 10 10 5 >> ../results/all_tables.csv

echoerr "################ correction - NN2 - AR" ###NN1 -> NN2
python3 main.py --dataset=correction --classifier=nn --method=AR --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv
echoerr "################ correction - NN2 - CF" ###NN1 -> NN2
python3 main.py --dataset=correction --classifier=nn --method=CF --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv
echoerr "################ correction - NN2 - Causal" ###NN1 -> NN2
python3 main.py --dataset=correction --classifier=nn --method=c12 --comment=COM --arch 20 10 10 10 5 >> ../results/all_tables.csv

