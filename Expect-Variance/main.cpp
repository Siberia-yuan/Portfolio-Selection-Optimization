#include <iostream>
#include "QuadProg++.hh"
#include <fstream>
#include <math.h>


int main (int argc, char *const argv[]) {
    float* portFolio(float **prices,float *weight,float sharpe,int rows,int cols);
    float* calMean(float **rets,int rows,int cols);
    float** calCov(float *means,float **rets,int rows,int cols);
    float calVar(float *weight,float **covMatrix,int rows,int cols);
    float calExpectation(float *weight,float *means,int cols);
    float minMean(float* Mean,int length);
    float maxMean(float* Mean,int length);
    float calExpect(float *Mean,float *weight,int cols);
    
    
    std::ifstream infile1;
    std::ifstream infile2;
    infile1.open("finantialData.txt");
    infile2.open("tickers.txt");
    int countTickers=0;
    std::string scapeGoat;
    
    while(getline(infile2,scapeGoat)){
        countTickers++;
    }
    infile2.close();
    
    
    int numLines=0;
    float receptor;
    while(!infile1.eof()){
        infile1>>receptor;
        numLines++;
    }
    infile1.close();
    numLines/=countTickers;

    
    float **ret;
    ret=new float*[numLines];
    for(int i_1=0;i_1<numLines;i_1++){
        ret[i_1]=new float[countTickers];
    }
    std::ifstream record;
    record.open("finantialData.txt");
    for(int frows=0; frows<numLines; frows++) {
        for(int fcols=0;fcols<countTickers;fcols++){
            record>>ret[frows][fcols];
        }
    }
    record.close();
    int rows=numLines;
    int cols=countTickers;
    
    
    float *mean=calMean(ret, rows, cols);
    float **cov=calCov(mean, ret, rows, cols);
    
    std::cout<<minMean(mean, cols)<<" "<<maxMean(mean, cols)<<std::endl;
    
    
    //matlab的最优解
    float *weight3;
    weight3=new float[14];
    weight3[0]=0.0417f;
    weight3[1]=0;
    weight3[2]=0.3046f;
    weight3[3]=0.0082f;
    weight3[4]=0.1195f;
    weight3[5]=0;
    weight3[6]=0;
    weight3[7]=0;
    weight3[8]=0;
    weight3[9]=0.396f;
    weight3[10]=0;
    weight3[11]=0.13f;
    weight3[12]=0;
    weight3[13]=0;
    std::cout<<"matlab var:"<<calVar(weight3, cov, rows, cols)<<std::endl;
    
    quadprogpp::Matrix<double> G, CE, CI;
    quadprogpp::Vector<double> g0, ce0, ci0, x;
    int n, m, p;
    
    n = cols;
    G.resize(n, n);
    for (int i_8 = 0; i_8 < n; i_8++){
        for (int i_9 = 0; i_9 < n; i_9++){
            G[i_8][i_9]=cov[i_8][i_9];
        }
    }
    
    g0.resize(n);
    for (int j_1=0; j_1<cols; j_1++) {
        g0[j_1]=0;
    }

    float exp=0.00178234;
    m = 2;
    CE.resize(n, m);
    for(int j_2=0;j_2<n;j_2++){
        for(int j_3=0;j_3<m;j_3++){
            if(j_3==0){
                CE[j_2][j_3]=1;
            }
            else{
                CE[j_2][j_3]=mean[j_2];
            }
        }
    }
    ce0.resize(m);
    
    ce0[0]=-1;
    ce0[1]=-exp;
    p = 2*cols;
    CI.resize(n, p);
    
    for(int j_5=0;j_5<n;j_5++){
        for(int j_6=0;j_6<cols;j_6++){
            if(j_5==j_6){
                CI[j_5][j_6]=1;
            }else{
                CI[j_5][j_6]=0;
            }
        }
        for(int j_7=cols;j_7<p;j_7++){
            if(j_5+cols==j_7){
                CI[j_5][j_7]=-1;
            }else{
                CI[j_5][j_7]=0;
            }
        }
    }
    ci0.resize(p);
    for(int j_8=0;j_8<cols;j_8++){
        ci0[j_8]=0;
    }
    for(int j_9=cols;j_9<p;j_9++){
        ci0[j_9]=1;
    }
    x.resize(n);
    
    
    std::cout << "optimize var: " << 2*solve_quadprog(G, g0, CE, ce0, CI, ci0, x) << std::endl;
    float* weight;
    weight=new float[cols];
    for(int k_0=0;k_0<cols;k_0++){
        std::cout<<x[k_0]<<std::endl;
        weight[k_0]=x[k_0];
    }
    std::cout << "expectation: " << calExpect(mean, weight, cols)<<std::endl;
    return 0;
}
float calExpectation(float *weight,float *means,int cols){
    float exp=0;
    for(int i=0;i<cols;i++){
        exp+=weight[i]*means[i];
    }
    return exp;
    
}
float* calMean(float **rets,int rows,int cols){
    //return to means of all the stocks
    float *means=new float[cols];
    for(int i=0;i<cols;i++){
        means[i]=0;
    }
    for(int j=0;j<cols;j++){
        for(int k=0;k<rows;k++){
            means[j]+=rets[k][j];
        }
    }
    for(int l=0;l<cols;l++){
        means[l]/=rows;
    }
    return means;
}
float**  calCov(float *means,float **rets,int rows,int cols){
    //return the matrix of covariance
    float** covMatrix=new float*[cols];
    for(int p=0;p<cols;p++){
        covMatrix[p]=new float[cols];
    }
    for(int i=0;i<cols;i++){
        for(int j=0;j<cols;j++){
            covMatrix[i][j]=0;
        }
    }
    for(int k=0;k<cols;k++){
        for(int j=0;j<cols;j++){
            for(int p=0;p<rows;p++){
                covMatrix[k][j]+=(rets[p][k]-means[k])*(rets[p][j]-means[j]);
            }
        }
    }
    for(int n=0;n<cols;n++){
        for(int m=0;m<cols;m++){
            covMatrix[n][m]/=rows-1;
        }
    }
    return covMatrix;
}
float calVar(float *weight,float **covMatrix,int rows,int cols){
    //return the overall variance
    float var=0;
    for(int i=0;i<cols;i++){
        for(int j=0;j<rows;j++){
            var+=weight[i]*weight[j]*covMatrix[i][j];
        }
    }
    return var;
}
float minMean(float *Mean,int Length){
    float minValue=Mean[0];
    for(int j=0;j<Length;j++){
        if(Mean[j]<minValue){
            minValue=Mean[j];
        }
    }
    return minValue;
}
float maxMean(float *Mean,int Length){
    float minValue=Mean[0];
    for(int j=0;j<Length;j++){
        if(Mean[j]>minValue){
            minValue=Mean[j];
        }
    }
    return minValue;
}
float calExpect(float *Mean,float *weight,int cols){
    float expect=0;
    for(int j=0;j<cols;j++){
        expect+=Mean[j]*weight[j];
    }
    return expect;
}

