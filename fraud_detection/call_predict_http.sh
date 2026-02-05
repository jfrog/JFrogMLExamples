#!/usr/bin/env bash

curl --location --request POST 'https://a0glmfje9adgj.ml.jfrog.io/v1/fraud_detection/default/predict' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer '$TOKEN'' \
--header 'X-JFrog-Tenant-Id: a0glmfje9adgj' \
--data '{
  "columns": ["V19","V3","V9","V5","V8","V6","V1","V4","V16","V17","V10","V18","V21","V7","V15","V27","Amount","V12","V22","V24","V26","V23","V28","V11","V2","V20","V14","V25","V13","Time"],
  "index": [0],
  "data": [[
    0.4576,
    -1.817515,
    0.336844,
    0.569073,
    -0.039657,
    -0.399897,
    2.0525,
    0.237443,
    0.624627,
    0.146194,
    -0.192936,
    0.35089,
    -0.338609,
    0.039466,
    -0.413996,
    -0.07046,
    1.79,
    0.448704,
    -0.931193,
    0.17091,
    0.174362,
    0.297277,
    -0.044793,
    0.773523,
    0.060449,
    -0.170036,
    -0.557454,
    -0.267115,
    -0.511748,
    123019.0
  ]]
}'