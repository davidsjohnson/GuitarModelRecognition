#!/usr/bin/env python

# bextract implemented using the swig python Marsyas bindings
# George Tzanetakis, January, 16, 2007 
# revised Graham Percival, 2011 August 20
# and again David Johnson, 2015 March 24

import sys
import marsyas
import marsyas_util

import numpy as np

def extractFeatures(collectionFile):


        # -> Memory{memSize=20}
        # -> Mean
    # Create top-level patch
    net = marsyas_util.create(
        ["Series/extract_network",
            ["SoundFileSource/src",
             "TimbreFeatures/featExtractor",
             "TextureStats/tStats",
             "Annotator/annotator"    # not doing anything with annotator but receive all zeros for tStats processedData without it?
        ]])

    # link the controls to coordinate things
    net.linkControl("mrs_string/filename",
        "SoundFileSource/src/mrs_string/filename")
    net.linkControl("mrs_bool/hasData",
        "SoundFileSource/src/mrs_bool/hasData")

    # set up features to extract
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableTDChild",
    #     marsyas.MarControlPtr.from_string("ZeroCrossings/zcrs"))
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableLPCChild",
    #     marsyas.MarControlPtr.from_string("Series/lspbranch"))
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableLPCChild",
    #     marsyas.MarControlPtr.from_string("Series/lpccbranch"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
        marsyas.MarControlPtr.from_string("MFCC/mfcc"))
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
    #     marsyas.MarControlPtr.from_string("SCF/scf"))
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
    #     marsyas.MarControlPtr.from_string("Rolloff/rf"))
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
    #     marsyas.MarControlPtr.from_string("Flux/flux"))
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
    #     marsyas.MarControlPtr.from_string("Centroid/cntrd"))
    # net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
    #     marsyas.MarControlPtr.from_string("Series/chromaPrSeries"))

    # setup filenames 
    net.updControl("mrs_string/filename",
        marsyas.MarControlPtr.from_string(collectionFile))

    # networkFile = open("extractNetwork.mpl", "w")
    # networkFile.write(net.toString())
    # networkFile.close()

    # do the processing, extracting features, and writing to weka file 
    previouslyPlaying = ""
    features = []
    labels = []
    sources = []
    while net.getControl("SoundFileSource/src/mrs_bool/hasData").to_bool():
        currentlyPlaying = net.getControl("SoundFileSource/src/mrs_string/currentlyPlaying").to_string()
        if (currentlyPlaying != previouslyPlaying):
            print "Processing: ",
            print net.getControl("SoundFileSource/src/mrs_string/currentlyPlaying").to_string()


        net.tick() # update time
        previouslyPlaying = currentlyPlaying

        # update arrays to store features
        features.append(net.getControl("TextureStats/tStats/mrs_realvec/processedData").to_realvec())
        labels.append(net.getControl("SoundFileSource/src/mrs_real/currentLabel").to_real())
        sources.append(currentlyPlaying)

    # get feature attributes and class labels
    #   feature attributes
    attrNamesStr = net.getControl("TextureStats/tStats/mrs_string/onObsNames").to_string()
    if attrNamesStr.endswith(",") : attrNamesStr = attrNamesStr[:-1]
    attrNames = attrNamesStr.split(",")

    #   class lables
    classesStr = net.getControl("SoundFileSource/src/mrs_string/labelNames").to_string()
    if classesStr.endswith(",") : classesStr = classesStr[:-1]
    classes = classesStr.split(",")

    # print np.array(features)
    return np.array(attrNames), np.array(features), np.array(labels), np.array(sources), np.array(classes)


if __name__ == '__main__':
    '''if called as executable and not a module import, write results
       to numpy binary file'''

    try:
        mf_filename = sys.argv[1]
    except:
        print "Usage: py-bextract.py filename.mf"
        sys.exit(1)


    # Write results to Numpy .NPZ binary file
    featureNames, features, labels, sources, classes = extractFeatures(mf_filename)
    np.savez(mf_filename.replace("mf", "npz"), featureNames=featureNames, features=features, labels=labels, sources=sources, classes=classes)



    