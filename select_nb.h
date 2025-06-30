//
// Created by 宋 on 2024/10/5.
//

#pragma once
#include "incrementalLearner.h"
#include "xyDist.h"
#include "xxyDist.h"
#include "xxxyDist.h"

class select_nb : public IncrementalLearner
{
public:

    /**
     * @param argv Options for the NB classifier
     * @param argc Number of options for NB
     */
    select_nb(char*const*& argv, char*const* end);


    ~select_nb(void);
    std::vector<unsigned int> selects;
    void reset(InstanceStream &is);   ///< reset the learner prior to training
    void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
    void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
    void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
    bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
    void getCapabilities(capabilities &c);

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param inst The instance to be classified
     * @param classDist Predicted class probability distribution
     */
    virtual void classify(const instance &inst, std::vector<double> &classDist);


    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @return  The joint distribution for each individual x-value and the class
     */
    xyDist* getXyDist();


private:
    bool trainingIsFinished_; ///< true iff the learner is trained
    xyDist xyDist_;           ///< the xy distribution that NB learns from the instance stream and uses for classification
    xxyDist xxyDist_;
    xxxyDist xxxyDist_;
    std::string datasetName;
    int percent;
    std::vector<int> atts;
};

