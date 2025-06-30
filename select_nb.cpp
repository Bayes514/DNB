//
// Created by 宋 on 2024/10/5.
//

#include "select_nb.h"
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "correlationMeasures.h"
#include <cmath>
#include "fstream"

select_nb::select_nb(char*const*& argv, char*const* end) : xyDist_(), trainingIsFinished_(false)
{
    percent=100;
    name_ = "select_nb";
        while (argv != end) {
        if (*argv[0] != '-') {
            break;
        }
        else if (argv[0][1] == 'y') {
            getUIntFromStr(argv[0] + 2, percent, "y");
        }
        else {
            break;
        }
        name_ += argv[0];
        ++argv;
    }


}


select_nb::~select_nb(void)
{
}

void  select_nb::getCapabilities(capabilities &c){
    c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void select_nb::reset(InstanceStream &is) {
    /*初始数据结构空间*/
    xyDist_.reset(&is);
    xxyDist_.reset(is);
    xxxyDist_.reset(is);
    trainingIsFinished_ = false;
}

class miCmpClass {
public:
    miCmpClass(std::vector<float> *m) {
        mi = m;
    }
    bool operator() (CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }
private:
    std::vector<float> *mi;
};

void select_nb::train(const instance &inst) {
    /*进行数据统计*/
    xyDist_.update(inst);
    xxyDist_.update(inst);
    xxxyDist_.update(inst);
}


void select_nb::initialisePass() {
}


void select_nb::finalisePass() {
    int noCatAtts_=xyDist_.getNoCatAtts();
    std::vector<int>order; //？？order存放的是所有的属性结点
    std::vector<int>order2;
    std::vector<int>order3;
    for (CategoricalAttribute a = 0; a < xxyDist_.getNoCatAtts(); a++) {
        order.push_back(a);
        order2.push_back(a);
        order3.push_back(a);
    }
    std::vector<float> mi=std::vector<float>(xxyDist_.getNoCatAtts());
//    getMutualInformation(xxyDist_.xyCounts,mi);
    getMutualInformation_fix_k(xxyDist_.xyCounts,mi,percent/100.0);

    atts.clear();
    miCmpClass cmp(&mi);
    std::sort(order.begin(), order.end(), cmp);
    atts.push_back(order[0]);


    crosstab<float> cmi = crosstab<float>(noCatAtts_);
//    getMitwoAtt1(xxyDist_,cmi);
    getMitwoAtt1_fix_k(xxyDist_,cmi,percent/100.0);
    miCmpClass cmp2(&cmi[order[0]]);
    std::sort(order2.begin(), order2.end(), cmp2);
    atts.push_back(order2[0]);


    std::vector<crosstab<float> > mi3=std::vector<crosstab<float> >(noCatAtts_,crosstab<float>(noCatAtts_));
    miCmpClass cmpadd(&mi3[order[0]][order2[0]]);

//    getMiallthreeAtt1(xxxyDist_,mi3);
    getMiallthreeAtt1_fix_k(xxxyDist_,mi3,percent/100.0);
    std::sort(order3.begin(), order3.end(), cmpadd);


    atts.push_back(order3[0]);
    std::cout<<"first three atts :"<<atts[0]<<" "<<atts[1]<<" "<<atts[2]<<"\n";


//    待选属性集合
    for(int o=0;o<noCatAtts_-3;o++)
    {
        std::vector<int> wait2add;
        std::vector<float> socre_wait2add;
        for(int i=0;i<noCatAtts_;i++)
            if(!std::count(atts.begin(),atts.end(),i)) {
                wait2add.push_back(i);
            }
        std::cout<<"----------------\n";
        int candidit=-1;
        float maxmi=-999;
        for(int x=0;x<wait2add.size();x++)
        {
            bool flag=false;
            float score=0;
//            std::cout<<wait2add[x]<<" : ";
            for(int ii=0;ii<atts.size();ii++)for(int jj=0;jj<atts.size();jj++){
                    int i=atts[ii];
                    int j=atts[jj];
                    if(i<=j) continue;
                    if(mi3[i][j][wait2add[x]]-cmi[i][j]<0)
                    {
                        flag=true;
                        break;
                    }
                    score+=mi3[i][j][wait2add[x]]-cmi[i][j];
//                    std::cout<<mi3[i][j][wait2add[x]]-cmi[i][j]<<"+";
                }
//            score+=mi[x];
//            std::cout<<"\t = "<<score<<"\n";
            if(flag)continue;
            if(score>maxmi)
            {
                maxmi=score;
                candidit=wait2add[x];
            }
            socre_wait2add.push_back(score);
        }
        if(candidit==-1)
            break;
        atts.push_back(candidit);
    }




    for(int i=0;i<atts.size();i++)
        std::cout<<atts[i]<<" ";


    trainingIsFinished_ = true;
}


bool select_nb::trainingIsFinished() {

    return trainingIsFinished_;
}

/*
 * 该函数是每次单独对一个实例进行分类
 */
void select_nb::classify(const instance &inst, std::vector<double> &classDist) {
  const unsigned int noClasses = xyDist_.getNoClasses();

  for (CatValue y = 0; y < noClasses; y++) {
    double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    // scale up by maximum possible factor to reduce risk of numeric underflow

    for (CategoricalAttribute a = 0; a < xyDist_.getNoAtts(); a++) {
        if(std::count(atts.begin(),atts.end(),a))
        p *= xyDist_.p(a, inst.getCatVal(a), y);
    }
    /*最后求得p就是c(inst)*/

    assert(p >= 0.0);
    /*classDist[y]存储的是实例属于每种类值的概率*/
    classDist[y] = p;
  }

  /*标准化类值概率，不影响类值概率大小关系*/
  normalise(classDist);
}

//void select_nb::classify(const instance &inst, std::vector<double> &classDist) {
//    int N=xyDist_.count;
//    std::vector<float> ce=std::vector<float>(xyDist_.getNoCatAtts());
//    getYCondEntropy(xyDist_,ce);
//    for(int i=0;i<xyDist_.getNoCatAtts();i++)
//    {
//        std::cout<<inst.getCatVal(i)<<" ";
//    }
//    std::cout<<"\n";
//    const unsigned int noClasses = xyDist_.getNoClasses();
//    double YEntropy=getYEntropy(xyDist_);
//    for (CatValue y = 0; y < noClasses; y++) {
////        double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
////        double p = pow(xyDist_.p(y),xyDist_.p(y)) ;
////        double p =xyDist_.p(y);
//        double p =pow(xyDist_.p(y),1-xyDist_.p(y));
//        std::cout<<p<<"\n";
//        for (unsigned int a = 0; a <  xyDist_.getNoAtts(); a++) {
//            std::cout<<" count(x"<<a+1<<",y"<<y+1<<")"<<" :"<<xyDist_.getCount(a,inst.getCatVal(a),y)<<std::endl;
//            double xiYEntropy= getXiYEntropy(xyDist_,a,inst.getCatVal(a));
////            std::cout<<"H(Y|xi) : "<<xiYEntropy<<"  "<<"H(Y) : "<<YEntropy<<std::endl;
//            //std::cout<<"P(y"<<y+1<<"|xi):"<<xyDist_.jointP(a,inst.getCatVal(a),y)/xyDist_.p(a,inst.getCatVal(a))<<"  "<<"P(y"<<y+1<<"):"<<xyDist_.p(y)<<" "<<xyDist_.jointP(a,inst.getCatVal(a),y)/(xyDist_.p(a,inst.getCatVal(a))*xyDist_.p(y))<<" "<<xyDist_.jointP(a,inst.getCatVal(a),y)<<" "<<pow(xyDist_.jointP(a,inst.getCatVal(a),y)/(xyDist_.p(a,inst.getCatVal(a))*xyDist_.p(y)),xyDist_.jointP(a,inst.getCatVal(a),y))<<std::endl;
////            std::cout<<"p(xi|y): "<<xyDist_.p(a,inst.getCatVal(a),y)<<" jointP: "<<xyDist_.jointP(a,inst.getCatVal(a),y)<<" res: "<<pow(xyDist_.p(a,inst.getCatVal(a),y),-xyDist_.jointP(a,inst.getCatVal(a),y))<<std::endl;
//
//                std::cout<<"H(Y|X):"<<ce[a]<<std::endl;
//
////            if(xiYEntropy<YEntropy)
////            {
////                p*=xyDist_.jointP(a,inst.getCatVal(a),y)/(xyDist_.p(a,inst.getCatVal(a))*xyDist_.p(y));
////            }
////            else {
////                p *=1;
////            }
////                p*=pow(xyDist_.p(a,inst.getCatVal(a),y),-xyDist_.jointP(a,inst.getCatVal(a),y));
////
//            //p*=pow(xyDist_.p(a,inst.getCatVal(a),y),N*xyDist_.jointP(a,inst.getCatVal(a),y));
////                p*=pow(xyDist_.jointP(a,inst.getCatVal(a),y)/xyDist_.p(a,inst.getCatVal(a)),N*xyDist_.jointP(a,inst.getCatVal(a),y))/pow(xyDist_.p(y),N*xyDist_.p(y));
////                    p*=pow(xyDist_.p(a,inst.getCatVal(a),y),ce[a]);
//
//
////            p*=xyDist_.p(a,inst.getCatVal(a),y)*pow(1-xyDist_.p(a,inst.getCatVal(a),y),-xyDist_.jointP(a,inst.getCatVal(a),y));
////            p*=pow(1-xyDist_.p(a,inst.getCatVal(a),y),-xyDist_.jointP(a,inst.getCatVal(a),y));
//                p*=pow(xyDist_.jointP(a,inst.getCatVal(a),y)/xyDist_.p(a,inst.getCatVal(a)),1-xyDist_.jointP(a,inst.getCatVal(a),y))/pow(xyDist_.p(y),1-xyDist_.p(y));
//                std::cout<<"p(xi|y): "<<xyDist_.p(a,inst.getCatVal(a),y)<<"(1-a)^-x : "<<pow(1-xyDist_.p(a,inst.getCatVal(a),y),-xyDist_.jointP(a,inst.getCatVal(a),y))<<std::endl;
//
//
////            p*=pow(xyDist_.p(a,inst.getCatVal(a),y),1-xyDist_.jointP(a,inst.getCatVal(a),y));
////            p*=pow(xyDist_.p(a,inst.getCatVal(a),y),-xyDist_.jointP(a,inst.getCatVal(a),y));
//            std::cout<<p<<"\n";
////            p*=xyDist_.p(a,inst.getCatVal(a),y);
//        }
//        assert(p >= 0.0);
//        std::cout<<p<<"\n";
//        /*classDist[y]存储的是实例属于每种类值的概率*/
//        classDist[y] = p;
//        std::cout<<"classify------\n";
//        for(int i=0;i<classDist.size();i++)
//        {
//            std::cout<<classDist[i]<<" ";
//        }
//        std::cout<<"\n";
//    }
//    std::cout<<"\n";
//    /*标准化类值概率，不影响类值概率大小关系*/
//    normalise(classDist);
//}

