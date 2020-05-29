package com.dpthinker.mnistclassifier;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import com.dpthinker.mnistclassifier.classifier.BaseClassifier;
import com.dpthinker.mnistclassifier.model.ModelConfigFactory;

import java.io.FileNotFoundException;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private final static String TAG = "MainActivity";
    private ImageView mImgView;
    private TextView mTextView;
    private RadioGroup mRadioGroup;
    private static boolean mIsFloat = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImgView = findViewById(R.id.mnist_img);
        mImgView.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(intent, 1);
            }
        });
        mTextView = findViewById(R.id.tv_prob);

        mRadioGroup = findViewById(R.id.rg_model);
        mRadioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup radioGroup, int i) {
                if (i == R.id.rb_quant) {
                    mIsFloat = false;
                } else {
                    mIsFloat = true;
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            Uri uri = data.getData();
            try {
                Bitmap bitmap = BitmapFactory.decodeStream(this.getContentResolver().openInputStream(uri));

                mImgView.setImageBitmap(bitmap);

                String config;
                if (mIsFloat) {
                    config = ModelConfigFactory.FLOAT_SAVED_MODEL;
                } else {
                    config = ModelConfigFactory.QUANT_SAVED_MODEL;
                }

                BaseClassifier classifier = new BaseClassifier(config, this);
                String result = classifier.doClassify(bitmap);

                mTextView.setText(result);
            } catch (FileNotFoundException e) {
                Log.e(TAG, "Not found input image: " + uri.toString());
            } catch (IOException e) {
                Log.e(TAG, "Exception in init Classifier");
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}