from . import *
import fastcore.methods.scale_utils.PQ as PQutils

def computeDis(scale, queryData, data, group, queryIdx=None, plainGroup=False):

    if scale.distanceAlg == "brute":
              
        groupbyMax = computeDisBrute(queryData, data, group,plainGroup)
    elif scale.distanceAlg == "brutePerDim":
              
        groupbyMax = computeDivideDis(queryData, data, group)
    elif scale.distanceAlg == "convex":
              
        groupbyMax = computeDivideHullDis(queryData, data, group, scale.hull, scale.g_hull)
    elif scale.distanceAlg == "brutePQ":
              
        groupbyMax = computePQScan(queryData.reshape(1, -1), data, group, scale.ipq, scale.pq)
    elif scale.distanceAlg =="pqADC":
              
        groupbyMax = computePQGroupSum(queryData.reshape(1, -1), data, group, scale.ipq, scale.pq, scale.groupPQids)
    elif scale.distanceAlg == "pqSDC":
              
        if scale.dataGroupDis is not None:
            groupbyMax = scale.dataGroupDis[queryIdx,:]
        else:
            print("Really compute utility from scratch!")

            groupbyMax = PQutils.computeDataGroupDisSDC(scale, queryIdx, plainGroup, groupPQDis=scale.groupPQdis)
                  
                  
    return groupbyMax