/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Random;

import weka.core.*;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.Scanner;
import weka.classifiers.Classifier;

/**
 *
 * @author Geraldi Dzakwan
 */
public class TucilWeka {
    
    DataSource dataSource;
    Instances inputDataSet;
    Instances outputDataSet;
    Classifier j48Classifier;
    
    public TucilWeka() {
        j48Classifier = new J48();
    }
    
    public void loadData() throws FileNotFoundException {
//        String loadFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/iris.arff";
        String loadFilePath = "iris.arff";
        try {
            dataSource = new DataSource(loadFilePath);
            inputDataSet = dataSource.getDataSet();
            inputDataSet.setClassIndex(inputDataSet.numAttributes() - 1);

            //Ini kalo instances langsung
            //Instances inputDataSet = new Instances(new BufferedReader(new FileReader(loadFilePath)));            
            
            System.out.println(inputDataSet.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void saveData() {
        ArffSaver aSaver = new ArffSaver();
        aSaver.setInstances(outputDataSet);
//        String saveFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/saved.arff";
        String saveFilePath = "saved.arff";
        try {
            aSaver.setFile(new File(saveFilePath));
            aSaver.writeBatch();
        } catch (IOException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void discretizeData() {
        int numberOfSettings = 3;
        numberOfSettings*=2;
        String[] settings = new String[numberOfSettings];
        
        String s1 = "-R", 
        s2 = "-B", 
        s3 = "-M";
        
        String v1 = "", v2 = "", v3 = "";
//        Scanner sc = new Scanner(System.in);
        
//        System.out.print("-R : ");
//        v1 = sc.nextLine();
//        
//        System.out.print("-B : ");
//        v2 = sc.nextLine();
//        
//        System.out.print("-M : ");
//        v3 = sc.nextLine();

        v1 = "first-last";
        v2 = "10";
        v3 = "-1";
        
        settings[0] = s1;
        settings[1] = v1;
        settings[2] = s2;
        settings[3] = v2;
        settings[4] = s3;
        settings[5] = v3;
        
        Discretize disc = new Discretize();
        try {
            disc.setOptions(settings);
            disc.setInputFormat(inputDataSet);
            outputDataSet = Filter.useFilter(inputDataSet, disc);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /*
        Function to train the classifiers
    */
    public void train() {
        try {
            j48Classifier.buildClassifier(inputDataSet);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void tenFoldEvaluation () {
        try {
            J48 tree = new J48();
            
            Evaluation eval = new Evaluation(inputDataSet);
            eval.crossValidateModel(tree, inputDataSet, 10, new Random(1));
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void fullTrainingEvaluation () {
        try {
            Instances test = inputDataSet;
            
            Evaluation eval = new Evaluation(inputDataSet);
            
            eval.evaluateModel(j48Classifier, test);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    } 
    
    /*
       Function to request a new instances from user using System.in
    */
    public Instance askInstancesFromUser() {
        int nAttributes = inputDataSet.numAttributes();
        Instance inst = new DenseInstance(nAttributes);
        Scanner s = new Scanner(System.in);
        String in;
        System.out.println(nAttributes);
        
        for (int i = 1; i <= nAttributes - 1; i++) {
            Attribute a = inputDataSet.attribute(i - 1);
            in = s.nextLine();
            if (a.isNominal()) { //predetermined nominal, ex: full, empty, some
                inst.setValue(a, in);
            } else if (a.isNumeric()) { //real values, ex: 5.2, 3.1
                inst.setValue(a, Float.parseFloat(in));
            }
        }
        
        return inst;
    }
    
    public void saveModels(String path) {
        try {
            SerializationHelper.write(path, j48Classifier);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void loadModels(String path) {
        try {
            j48Classifier = (Classifier) weka.core.SerializationHelper.read(path);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void classifyInstance(Instance inst) {
        try {
            double classLabel = j48Classifier.classifyInstance(inst);
            inst.setClassValue(classLabel);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        TucilWeka test = new TucilWeka();
        
        try {
            test.loadData();

            test.discretizeData();
            
            test.train();
            
            //pilih either fulltraining atau 10-fold
            
            //buat hipotesis
            
            //tampilin hipotesis
            
            //simpan hipotesis
            
            //baca instans baru
            
            //klasifikasi
            
            test.saveData();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
