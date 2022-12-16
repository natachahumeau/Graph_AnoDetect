#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include "time.h"

#include "accuracy.cpp"
#include "anomaly_detect.cpp"
#include "anomaly_inject.cpp"
#include "pagerank.cpp"
#include "read_data.cpp"

#define attackLimit 1

using namespace std;

int main(int argc, char *argv[])
{
    clock_t start = clock();

    string path = argv[1];
    string delimeter = argv[2];
    int timeStep = atoi(argv[3]);
    int initSS = atoi(argv[4]);
    int injectScene = atoi(argv[5]);
    int injectNum = atoi(argv[6]);
    int injectSize = atoi(argv[7]);
    bool INJECT = (injectScene != 0);

    // READ DATA
    vector<timeEdge> edges;
    vector<int> snapshots;
    vector<int> labels;
    int n, m, timeNum;
    read_data(path, delimeter, timeStep, edges, snapshots, n, m, timeNum, labels);
    int numSS = timeNum/timeStep + 1;
    outEdge* A = new outEdge[n];
    cout << "#node: " << n << ", #edges: "<< edges.size() << ", #timeStamp: " << timeNum << ", #numSS: " << numSS << endl;

    // ANOMALY_SCORE
    int testNum = numSS - initSS;
    bool* attack = new bool[testNum];
    double* anomScore = new double[testNum];
    for(int i = 0; i < testNum; i++)
    {
        anomScore[i] = 0;
        attack[i] = false;
    }

    // PAGERANK
    double** pagerank1 = new double*[3];
    double** pagerank2 = new double*[3];
    for(int i = 0; i < 3; i++)
    {
        pagerank1[i] = new double[n];
        pagerank2[i] = new double[n];
        for(int j = 0; j < n; j++)
            pagerank1[i][j] = pagerank2[i][j] = 0;
    }

    // MEAN AND VARIANCE
    double** mean = new double*[4];
    double** var = new double*[4];
    for(int i = 0; i < 4; i++)
    {
        mean[i] = new double[n];
        var[i] = new double[n];
        for(int j = 0; j < n; j++)
            mean[i][j] = var[i][j] = 0;
    }

    // INJECTED SNAPSHOT
    vector<int> injectSS;
    if(INJECT)
        inject_snapshot(injectNum, initSS, testNum, snapshots, injectSS);

    cout << "Preprocess done: " << (double)(clock() - start) / CLOCKS_PER_SEC << endl;

    int eg = 0;
    int snapshot = 0;
    int attackNum = 0;
    int injected = 0;
    int current_m = 0;
    double previous_score = 100.0;

    start = clock();
    int print_e = 10;

    int tp, fp, tn, fn;
    tp = fp = tn = fn = 0;

    for(int ss = 0; ss < snapshots.size(); ss++)
    {
        //std::cout << "snap = " << snapshots[ss] << std::endl;
        while(edges[eg].t < snapshots[ss]*timeStep)
        {
            inject(A, edges[eg].src, edges[eg].trg, 1);
            current_m++;
            if(edges[eg].atk)
                attackNum++;
            eg++;
            if(eg == print_e)
            {
                cout << eg << "," << (double)(clock() - start) / CLOCKS_PER_SEC << endl;
                print_e *= 10;
            }
            if(eg == edges.size())
                break;
        }

        if(INJECT && injectSS[injected] == snapshots[ss])
        {
            current_m += inject_anomaly(injectScene, A, n, injectSize);
            attackNum += attackLimit;
            injected++;
            if(injected == injectSS.size())
                INJECT = false;
        }

        snapshot = snapshots[ss];
        pagerank(A, pagerank1[snapshot%3], n, current_m, 1);
        pagerank(A, pagerank2[snapshot%3], n, current_m, 2);

        double score = compute_anomaly_score(snapshot, pagerank1, pagerank2, mean, var, n);
        if(snapshot >= initSS)
        {
            anomScore[snapshot - initSS] = score; //min(score, previous_score);
            attack[snapshot - initSS] = attackNum >= attackLimit;
            //std::cout << "attack = " << attackNum << std::endl;
            //std::cout << "attack = " << snapshot - initSS << std::endl;
            previous_score = score;

            /*
            int indx = snapshot - initSS;
            if(attack[indx] == 1) {
                if (attack[indx] == labels[indx]) 
                    tp ++;
                else 
                    fp ++;
            }
            else {
                if (attack[indx] == labels[indx]) 
                    tn ++;
                else {
                    fn ++;
                    //std::cout << "attack num = " << attackNum << std::endl;
                }
            
            };
            */
            /*
            if (indx%100) {
                //std::cout << "TP:" << tp << "; FP:" << fp << "; TN:" << tn << "; FN:" << fn << std::endl;
                //std::cout << "attack num = " << attackNum << std::endl;
            }
            */
        }
        attackNum = 0;
    }

    
    // WRITE ANOMALY SCORE
    string filePath = "UCI_anomrank.txt";  //"data_test2_anomrank.txt";
    ofstream writeFile;
    writeFile.open(filePath.c_str(), ofstream::out);
    for(int i = 0; i < testNum; i++)
        writeFile << anomScore[i] << " " << int(attack[i]) << endl;
    writeFile.close();

    /*
    double prec = 0;
    double recal = 0;
    double tpr = 0;
    double fpr = 0;
    int lg = 160;
    int mult = 20;
    double* precisions = new double[lg];
    double* recalls = new double[lg];
    double* FPRs = new double[lg];
    double* TPRs = new double[lg];
    
    // COMPUTE ACCURACY
    for(int i = 1; i < lg; i ++) {
        compute_accuracy(anomScore, attack, testNum, mult*i, &prec, &recal, &tpr, &fpr);
        precisions[i] = prec;
        recalls[i] = recal;
        TPRs[i] = tpr;
        FPRs[i] = fpr;
    }
    */
    /*
    // Write Accuracies resuslts
    string filePath2 = "darpa_anomrank_accuracies2.txt"; //"data_test2_anomrank_accuracies.txt";
    ofstream writeFile2;
    writeFile2.open(filePath2.c_str(), ofstream::out);
    for(int i = 0; i < lg; i++)
        writeFile2 << precisions[i] << " " << recalls[i] << " " << TPRs[i] << " " << FPRs[i] << endl;
    writeFile2.close();

    std::cout << "Final : TP:" << tp << "; FP:" << fp << "; TN:" << tn << "; FN:" << fn << std::endl;

    int som = 0;
    for (int i = 0; i < labels.size(); i++) {
        if (labels[i] == 1) 
            som++;
    }
    std::cout << "som" << som << "tot" << labels.size() << std::endl;

    tp = fp = tn = fn = 0;
    int topN = lg*mult;
    std::vector<size_t> idx(testNum);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&anomScore](size_t i1, size_t i2) {return anomScore[i1] > anomScore[i2];});

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

    std::cout << "Other Final : TP:" << tp << "; FP:" << fp << "; TN:" << tn << "; FN:" << fn << std::endl;
    */

    // FREE MEMORY
    delete [] A;
    delete [] anomScore;

    for(int i = 0; i < 3; i++)
    {
       delete [] pagerank1[i];
       delete [] pagerank2[i];
    }
    delete [] pagerank1;
    delete [] pagerank2;

    for(int i = 0; i < 4; i++)
    {
        delete [] mean[i];
        delete [] var[i];
    }
    delete [] mean;
    delete [] var;

    return 1;
}

