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

    # Create top-level patch
    net = marsyas_util.create(
        ["Series/extract_network",
            ["SoundFileSource/src",
             "TimbreFeatures/featExtractor",
             "TextureStats/tStats",
             "Annotator/annotator",
        ]])

    # link the controls to coordinate things
    net.linkControl("mrs_string/filename",
        "SoundFileSource/src/mrs_string/filename")
    net.linkControl("mrs_bool/hasData",
        "SoundFileSource/src/mrs_bool/hasData")
    net.linkControl("Annotator/annotator/mrs_real/label",
        "SoundFileSource/src/mrs_real/currentLabel")

    # set up features to extract
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableTDChild",
        marsyas.MarControlPtr.from_string("ZeroCrossings/zcrs"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableLPCChild",
        marsyas.MarControlPtr.from_string("Series/lspbranch"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableLPCChild",
        marsyas.MarControlPtr.from_string("Series/lpccbranch"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
        marsyas.MarControlPtr.from_string("MFCC/mfcc"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
        marsyas.MarControlPtr.from_string("SCF/scf"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
        marsyas.MarControlPtr.from_string("Rolloff/rf"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
        marsyas.MarControlPtr.from_string("Flux/flux"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
        marsyas.MarControlPtr.from_string("Centroid/cntrd"))
    net.updControl("TimbreFeatures/featExtractor/mrs_string/enableSPChild",
        marsyas.MarControlPtr.from_string("Series/chromaPrSeries"))

    # setup filenames 
    net.updControl("mrs_string/filename",
        marsyas.MarControlPtr.from_string(collectionFile))

    networkFile = open("extractNetwork.mpl", "w")
    networkFile.write(net.toString())
    networkFile.close()

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
        labels.append(net.getControl("Annotator/annotator/mrs_real/label").to_real())
        sources.append(currentlyPlaying)

    classes = net.getControl("SoundFileSource/src/mrs_string/labelNames").to_string()
    if classes.endswith(",") : classes = classes[:-1]
    classes = classes.split(",")
    return np.array(features), np.array(labels), np.array(sources), np.array(classes)


## Unit Test
if __name__ == '__main__':

    try:
        mf_filename = sys.argv[1]
    except:
        print "Usage: py-bextract.py filename.mf"
        sys.exit(1)


    features, labels, sources, classes = extractFeatures(mf_filename)
    np.savez(mf_filename.replace("mf", "npz"), features=features, labels=labels, sources=sources, classes=classes)