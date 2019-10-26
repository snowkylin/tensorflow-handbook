package com.dpthinker.mnistclassifier.model;

import java.nio.ByteBuffer;

public class QuantSavedModelConfig extends BaseModelConfig {
    @Override
    protected void setConfigs() {
        setModelName("mnist_savedmodel_quantized.tflite");

        setNumBytesPerChannel(4);

        setDimBatchSize(1);
        setDimPixelSize(1);

        setDimImgWeight(28);
        setDimImgHeight(28);

        setImageMean(0);
        setImageSTD(255.0f);
    }

    @Override
    public void addImgValue(ByteBuffer imgData, int val) {
        imgData.putFloat(((val & 0xFF) - getImageMean()) / getImageSTD());
    }
}
