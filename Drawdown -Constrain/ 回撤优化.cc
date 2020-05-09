#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string.h>
#include <vector>
#include <math.h>
#include <limits>
#include <set>
#include <map>
#include "ortools/base/commandlineflags.h"
#include "ortools/base/commandlineflags.h"
#include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#define constrain 0.15
int cols,rows;
float **cumuSum;
float *mean;
namespace operations_research {
    void LinearProgrammingExample() {
        MPSolver solver("LinearExample", MPSolver::GLOP_LINEAR_PROGRAMMING);
        // [END solver]
        
        // [START variables]
        const double infinity = solver.infinity();
        // x and y are non-negative variables.
        //MPVariable* const x = solver.MakeNumVar(0.0, infinity, "x");
        MPVariable** x;
        x=new MPVariable*[cols];
        for(int j1=0;j1<cols;j1++){
            x[j1]=solver.MakeNumVar(0.0,1,std::to_string(j1));
        }
        //MPVariable* const y = solver.MakeNumVar(0.0, infinity, "y");
        MPVariable** y;
        y=new MPVariable*[rows+1];
        for(int j2=0;j2<rows+1;j2++){
            y[j2]=solver.MakeNumVar(-infinity,infinity,std::to_string(j2+cols));
        }
        LOG(INFO) << "Number of variables = " << solver.NumVariables();
        // [END variables]
        
        // [START constraints]
        // x + 2*y <= 14.
        /*
         MPConstraint* const c0 = solver.MakeRowConstraint(-infinity, 14.0);
         c0->SetCoefficient(x, 1);
         c0->SetCoefficient(y, 2);
         */
        MPConstraint* const c0 = solver.MakeRowConstraint(1, 1);
        for(int j5=0;j5<cols;j5++){
            c0->SetCoefficient(x[j5], 1);
        }
        for(int j6=0;j6<rows+1;j6++){
            c0->SetCoefficient(y[j6], 0);
        }
        
        MPConstraint* const c1 = solver.MakeRowConstraint(0, 0);
        for(int j7=0;j7<cols;j7++){
            c1->SetCoefficient(x[j7], 0);
        }
        c1->SetCoefficient(y[0],1);
        for(int j8=1;j8<rows+1;j8++){
            c1->SetCoefficient(y[j8], 0);
        }
        
        MPConstraint** c;
        c=new MPConstraint*[3*rows];
        for(int k1=0;k1<rows;k1++){
            c[k1]=solver.MakeRowConstraint(-infinity,0);
            for(int o1=0;o1<cols;o1++){
                c[k1]->SetCoefficient(x[o1],0);
            }
            for(int o2=0;o2<1+rows;o2++){
                if(o2==k1){
                    c[k1]->SetCoefficient(y[o2],1);
                }
                if(o2==k1+1){
                    c[k1]->SetCoefficient(y[o2],-1);
                }
                if(o2!=k1+1&&o2!=k1){
                    c[k1]->SetCoefficient(y[o2],0);
                }
            }
        }
        
        for(int k2=rows;k2<2*rows;k2++){
            c[k2]=solver.MakeRowConstraint(-infinity,0);
            for(int o3=0;o3<cols;o3++){
                c[k2]->SetCoefficient(x[o3],cumuSum[k2-rows][o3]);
            }
            for(int o4=0;o4<1+rows;o4++){
                if(k2==o4+rows-1){
                    c[k2]->SetCoefficient(y[o4],-1);
                }else{
                    c[k2]->SetCoefficient(y[o4],0);
                }
            }
        }
        
        for(int k3=2*rows;k3<3*rows;k3++){
            c[k3]=solver.MakeRowConstraint(-infinity,constrain);
            for(int o5=0;o5<cols;o5++){
                c[k3]->SetCoefficient(x[o5],-1*cumuSum[k3-2*rows][o5]);
            }
            for(int o6=0;o6<1+rows;o6++){
                if(k3==o6+2*rows-1){
                    c[k3]->SetCoefficient(y[o6],1);
                }else{
                    c[k3]->SetCoefficient(y[o6],0);
                }
            }
        }
        
        
        LOG(INFO) << "Number of constraints = " << solver.NumConstraints();
        // [END constraints]
        
        // [START objective]
        // Objective function: 3x + 4y.
        MPObjective* const objective = solver.MutableObjective();
        for(int j3=0;j3<cols;j3++){
            objective->SetCoefficient(x[j3],mean[j3]);
        }
        for(int j4=0;j4<rows+1;j4++){
            objective->SetCoefficient(y[j4],0);
        }
        objective->SetMaximization();
        // [END objective]
        
        // [START solve]
        const MPSolver::ResultStatus result_status = solver.Solve();
        // Check that the problem has an optimal solution.
        if (result_status != MPSolver::OPTIMAL) {
            LOG(FATAL) << "The problem does not have an optimal solution!";
        }
        // [END solve]
        
        // [START print_solution]
        LOG(INFO) << "Solution:";
        LOG(INFO) << "Optimal objective value = " << objective->Value();
        //LOG(INFO) << x->name() << " = " << x->solution_value();
        //LOG(INFO) << y->name() << " = " << y->solution_value();
        for(int jk1=0;jk1<cols;jk1++){
            LOG(INFO)<<x[jk1]->solution_value();
        }
        // [END print_solution]
    }
}  // namespace operations_research

// namespace operations_research

int main(int argc, char** argv) {
    //float* portFolio(float **prices,float *weight,float sharpe,int rows,int cols);
    float* calMean(float **rets,int rows,int cols);
    float** calCov(float *means,float **rets,int rows,int cols);
    float calVar(float *weight,float **covMatrix,int rows,int cols);
    float calExpectation(float *weight,float *means,int cols);
    float minMean(float* Mean,int length);
    float maxMean(float* Mean,int length);
    float calExpect(float *Mean,float *weight,int cols);
    float** cumu(float **rets,int rows,int cols);
    
    
    std::ifstream infile1;
    std::ifstream infile2;
    infile1.open("finantialData.txt");
    infile2.open("tickers.txt");
    int countTickers=0;
    std::string scapeGoat;
    //这里缺了一个文件异常处理的部分，需要做一点改进
    while(getline(infile2,scapeGoat)){
        countTickers++;
    }
    
    infile2.close();
    //storing tickers
    std::ifstream readLabel;
    readLabel.open("tickers.txt");
    std::string tickers[countTickers];
    for(int i_0=0;i_0<countTickers;i_0++){
        getline(readLabel,tickers[i_0]);
    }
    int numLines=0;
    float receptor;
    while(!infile1.eof()){
        infile1>>receptor;
        numLines++;
    }
    infile1.close();
    //std::cout<<numLines<<std::endl;
    //std::cout<<countTickers<<std::endl;
    numLines/=countTickers;
   
    //std::cout<<numLines<<std::endl;
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
    rows=numLines;
    cols=countTickers;
    //float **ret=price2ret(prices, rows+1, cols);
    mean=calMean(ret, rows, cols);
    float **cov=calCov(mean, ret, rows, cols);
    cumuSum=cumu(ret, rows, cols);
    

    operations_research::LinearProgrammingExample();
    return EXIT_SUCCESS;
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

float **cumu(float **rets,int rows,int cols){
    float **cumuSum;
    cumuSum=new float*[rows];
    for(int i=0;i<rows;i++){
        cumuSum[i]=new float[cols];
    }
    float* records;
    records=new float[cols];
    for(int l=0;l<cols;l++){
        records[l]=0;
    }
    for(int j=0;j<rows;j++){
        for(int k=0;k<cols;k++){
            records[k]+=rets[j][k];
            cumuSum[j][k]=records[k];
        }
    }
    return cumuSum;
}
