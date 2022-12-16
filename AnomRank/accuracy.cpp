#include <vector>
#include <iostream>

void compute_accuracy(double* as, bool* attack, int timeNum, int topN, double* prec, double* recal, double* TPR, double* FPR)
{
    int tp, fp, tn, fn;
    tp = fp = tn = fn = 0;
    std::vector<size_t> idx(timeNum);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&as](size_t i1, size_t i2) {return as[i1] > as[i2];});

    //for (int i = 0; i < idx.size(); i++) {
    //    std::cout << idx[i] << ' ';
    //}

    for(int i = 0; i < topN; i++)
    {
        if(attack[idx[i]])
            tp++;
        else
            fp++;
	}

    for(int i = topN; i < timeNum; i++)
    {
        if(attack[idx[i]])
            fn++;
        else
            tn++;
    }

    double precision = double(tp)/(tp+fp);
    double recall = double(tp)/(tp+fn);
    double f1 = 2*precision*recall/(precision+recall);
    *TPR = double(tp)/(tp+fn);
    *FPR = double(fp)/(fp+tn);
    *prec = precision;
    *recal = recall;

    std::cout << "[TOP" << topN << "] precision: " << precision << ", recall: " << recall << std::endl;
    //if (topN % 100 ) {
    //    std::cout << "TP:" << tp << "; FP:" << fp << "; TN:" << tn << "; FN:" << fn << std::endl;
    //}
}



