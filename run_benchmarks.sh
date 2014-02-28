declare -a algolist=(rand anneal tpe tree gp_tree)
declare -a datalist=(newsgroups convex mnist)
declare -a seedlist=(1 2 3 4 5)
loc=~/hyperopt-sklearn-benchmarks
for data in ${datalist[@]}
do
    for algo in ${algolist[@]}
    do
        for seed in ${seedlist[@]}
        do
      	    python $loc/hpsklearn_benchmark.py -d $data -a $algo -s $seed -e 50 -t script_run_ &
        done
        wait
    done
    wait
done
