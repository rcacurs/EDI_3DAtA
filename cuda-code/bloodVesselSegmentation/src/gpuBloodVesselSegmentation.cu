#include"../include/gpuBloodVesselSegmentation.cuh"


ImMatG * extractFeatures(ImMatG *inputLayerG, ImMatG *dCodes, ImMatG *dMeans, cublasHandle_t handle){

	ImMatG *inputLayerT = inputLayerG->transpose();

	ImagePyramidCreator *creator = new ImagePyramidCreator(6, 5, 1);

	std::vector<ImMatG *> pyramid = creator->createPyramid(inputLayerT);

	//feature extraction

	ImMatG *features = new ImMatG(dCodes->rows*pyramid.size() + 1, inputLayerT->rows*inputLayerT->cols);
	delete inputLayerT;
	for (int i = 0; i < pyramid.size(); i++){

		cudaThreadSynchronize();

		ImMatG *paddedLayer = new ImMatG();
		padArray(pyramid[i], paddedLayer, 2);

		ImMatG *patches = im2col(paddedLayer, 5);
		ImMatG *integralG = computeIntegralImage(paddedLayer);
		ImMatG *integralSqG = computeIntegralImageSq(paddedLayer);
		delete paddedLayer;

		// compute means
		ImMatG *means = new ImMatG();

		meansOfPatches(integralG, 5, means);
		
		ImMatG *patchesNoMeans = new ImMatG();

		bsxfunminusCol(patches, patchesNoMeans, means->data_d);
		delete patches;
		delete means;
		cudaThreadSynchronize();



		//compute variances
		ImMatG *var = new ImMatG();
		variancesOfPatches(integralG, integralSqG, 5, var);
		delete integralSqG;
		delete integralG;

		ImMatG *patchesNorm = new ImMatG();
		bsxfundivide(patchesNoMeans, patchesNorm, var->data_d);
		delete patchesNoMeans;
		delete var;
		//delete patchesNorm;
		ImMatG *patchesNoFilterMean = new ImMatG();
		bsxfunminusRow(patchesNorm, patchesNoFilterMean, dMeans->data_d);
		delete patchesNorm;

		ImMatG *result = new ImMatG(dCodes->rows, patchesNoFilterMean->rows);

		double alf = 1;
		double bet = 0;
		const double *alfa = &alf;
		const double *beta = &bet;

		cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			patchesNoFilterMean->rows,
			dCodes->rows,
			patchesNoFilterMean->cols,
			alfa,
			patchesNoFilterMean->data_d,
			patchesNoFilterMean->cols,
			dCodes->data_d,
			dCodes->cols,
			beta,
			result->data_d, patchesNoFilterMean->rows);
		delete patchesNoFilterMean;

		for (int j = 0; j < dCodes->rows; j++){
			ImMatG * tempFeature = new ImMatG(inputLayerG->rows / pow(2.0, i), inputLayerG->rows / pow(2.0, i));


			if (i == 0){

				cudaMemcpy(tempFeature->data_d, &((result->data_d)[j*tempFeature->getLength()]), sizeof(double)*tempFeature->getLength(), cudaMemcpyDeviceToDevice);
				ImMatG * tempFeatureT = tempFeature->transpose();
				cudaMemcpy(&((features->data_d)[j*tempFeature->getLength()]), tempFeatureT->data_d, sizeof(double)*tempFeature->getLength(), cudaMemcpyDeviceToDevice);
				delete tempFeatureT;

			}
			else{

				cudaMemcpy(tempFeature->data_d, &((result->data_d)[j*tempFeature->getLength()]), sizeof(double)*tempFeature->getLength(), cudaMemcpyDeviceToDevice);
				ImMatG *tempResized = bicubicResize(tempFeature, pow(2.0, i));

				int featureIdx = j + i*(dCodes->rows);
				cudaMemcpy(&((features->data_d)[featureIdx*(tempResized->getLength())]), tempResized->data_d, sizeof(double)*tempResized->getLength(), cudaMemcpyDeviceToDevice);
				cudaThreadSynchronize();
				delete tempResized;
			}
			delete tempFeature;
		}
		delete result;

	}
	delete creator;
	clearPyramid(pyramid);
	pyramid.clear();
	cudaMemset(&((features->data_d)[((features->rows) - 1)*features->cols]), 1, sizeof(double)*features->cols);
	return features;

}



ImMatG * classify(ImMatG * features, int rows, int cols, ImMatG *scaleMean, ImMatG *model, ImMatG *scalesSd, cublasHandle_t handle){
	

	features->fillRow(192, 1);

	ImMatG * featuresNoMean = new ImMatG();

	bsxfunminusCol(features, featuresNoMean, scaleMean->data_d);

	ImMatG * featuresNoSd = new ImMatG();

	bsxfundivide(featuresNoMean, featuresNoSd, scalesSd->data_d);

	delete featuresNoMean;

	double alf = 1;
	double bet = 0;
	const double *alfa = &alf;
	const double *beta = &bet;
	ImMatG *multResult = new ImMatG(model->rows, featuresNoSd->cols);
	cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		featuresNoSd->cols,
		model->rows,
		featuresNoSd->rows,
		alfa,
		featuresNoSd->data_d,
		featuresNoSd->cols,
		model->data_d,
		model->cols,
		beta,
		multResult->data_d, multResult->cols);

	delete featuresNoSd;

	ImMatG * maxValues = new ImMatG();

	maxColumnVal(multResult, maxValues);

	ImMatG * resNoMax = new ImMatG();
	bsxfunminusRow(multResult, resNoMax, maxValues->data_d);

	delete maxValues;
	delete multResult;

	ImMatG * resExp = new ImMatG();

	expElem(resNoMax, resExp);

	delete resNoMax;
	ImMatG * resExpSum = new ImMatG();

	columnSum(resExp, resExpSum);


	ImMatG *res = new ImMatG();
	bsxfunRowDivide(resExp, res, resExpSum->data_d);
	delete resExp;
	delete resExpSum;

	ImMatG *segmRes = new ImMatG(rows, cols);
	cudaMemcpy(segmRes->data_d, &((res->data_d)[segmRes->getLength()]), segmRes->getLength()*sizeof(double), cudaMemcpyDeviceToDevice);
	delete res;
	return segmRes;
}

ImMatG *segmentBloodVessels(ImMatG *inputLayerG, ImMatG *dCodes, ImMatG *dMeans, ImMatG *scaleMean, ImMatG *model, ImMatG *scalesSd){
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	ImMatG *features = extractFeatures(inputLayerG, dCodes, dMeans, handle);
	ImMatG *segmRes = classify(features, inputLayerG->rows, inputLayerG->cols, scaleMean, model, scalesSd, handle);
	delete features;
	return segmRes;
}
