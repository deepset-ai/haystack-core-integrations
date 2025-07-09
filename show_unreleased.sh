#!/bin/bash
INTEGRATION=$1
if [ -z "${INTEGRATION}" ] ; then
    echo "Please provide the name of an integration, for example:"
    echo "./$(basename $0) chroma"
    exit 1
fi
LATEST_TAG=$(git tag -l --sort=-creatordate "integrations/${INTEGRATION}-v*" | head -n 1)
git --no-pager diff $LATEST_TAG..main integrations/${INTEGRATION}
