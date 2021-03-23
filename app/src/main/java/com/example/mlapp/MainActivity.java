package com.example.mlapp;

import androidx.appcompat.app.AppCompatActivity;

import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

import android.content.res.AssetManager;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;

import weka.core.Instances;

import android.os.Bundle;

public class MainActivity extends AppCompatActivity {
    TextView tv;
    Button b;
    EditText ed;
    final Attribute attributetsh = new Attribute("tsh");

    final List<String> classes = new ArrayList<String>() {
        {
            add("negative");
            add("hypothyroid");
        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        b = findViewById(R.id.b);
        tv=findViewById(R.id.tv);
        ed=findViewById(R.id.ed);

    }

    public void submit(View view) {

        double d= Double.parseDouble(ed.getText().toString());
        // Instances(...) requires ArrayList<> instead of List<>...
        ArrayList<Attribute> attributeList = new ArrayList<Attribute>(2) {
            {
                add(attributetsh);
                Attribute attributeClass = new Attribute("@@class@@", classes);
                add(attributeClass);
            }
        };
        // unpredicted data sets (reference to sample structure for new instances)
        Instances dataUnpredicted = new Instances("TestInstances",
                attributeList, 1);
        // last feature is target variable
        dataUnpredicted.setClassIndex(dataUnpredicted.numAttributes() - 1);

        // create new instance: this one should fall into the setosa domain
        DenseInstance newInstancePredict = new DenseInstance(dataUnpredicted.numAttributes()) {
            {
                setValue(attributetsh, d);
            }
        };

        // instance to use in prediction
        DenseInstance newInstance = newInstancePredict;
        // reference to dataset
        newInstance.setDataset(dataUnpredicted);

        // import ready trained model
        Classifier cls = null;
        try {
            AssetManager assetManager = getAssets();
            cls = (Classifier) weka.core.SerializationHelper
                    .read(assetManager.open("modeln.model"));
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (cls == null)
            return;

        // predict new sample
        try {
            double result = cls.classifyInstance(newInstance);
            System.out.println("Index of predicted class label: " + result );
            if(result==0.0)
                tv.setText("Hypothroid NEGATIVE");
            else if(result==1.0)
                tv.setText("Hypothyroid POSITIVE");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}



