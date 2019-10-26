package com.dpthinker.mnistclassifier.model;

public class ModelConfigFactory {
    public final static String FLOAT_SAVED_MODEL = "float_saved_model";
    public final static String QUANT_SAVED_MODEL = "quant_saved_model";

    public static BaseModelConfig getModelConfig(String model) {
        if (model == FLOAT_SAVED_MODEL) {
            return new FloatSavedModelConfig();
        } else if (model == QUANT_SAVED_MODEL) {
            return new QuantSavedModelConfig();
        }
        return null;
    }
}
