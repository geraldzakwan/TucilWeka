package tucilweka;

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Random;
import weka.core.*;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;

public class TucilWeka {
    DataSource dataSource;
    Instances inputDataSet;
    Instances filteredDataSet;
    Instances unsetData;
    Classifier classifier;
    int cType;
    Scanner sc;
    weka.filters.unsupervised.attribute.Discretize unSuperDisc;
    weka.filters.supervised.attribute.Discretize superDisc;
    
    public TucilWeka() {
        sc = new Scanner(System.in);
    }
    
    public void loadData(String loadFilePath) throws FileNotFoundException {
        try {
            dataSource = new DataSource(loadFilePath);
            inputDataSet = dataSource.getDataSet();
            inputDataSet.setClassIndex(inputDataSet.numAttributes() - 1);
            System.out.println(inputDataSet.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void saveData(String saveFilePath) {
        ArffSaver aSaver = new ArffSaver();
        aSaver.setInstances(inputDataSet);
        try {
            aSaver.setFile(new File(saveFilePath));
            aSaver.writeBatch();
        } catch (IOException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void discretizeData() {
        System.out.print("Supervised (1) or unsupervised (2) discretize : ");
        int numberOfSettings;
        String[] settings;
        if(sc.nextInt()==2) {
            numberOfSettings = 3;
            settings = new String[numberOfSettings*2];

            String s1 = "-R", 
            s2 = "-B", 
            s3 = "-M";
            String v1 = "first-last";
            String v2 = "10";
            String v3 = "-1";
            //System.out.print("-R : ");
            //v1 = sc.nextLine();
            //System.out.print("-B : ");
            //v2 = sc.nextLine();
            //System.out.print("-M : ");
            //v3 = sc.nextLine();

            settings[0] = s1;
            settings[1] = v1;
            settings[2] = s2;
            settings[3] = v2;
            settings[4] = s3;
            settings[5] = v3;
            
            unSuperDisc = new weka.filters.unsupervised.attribute.Discretize();
            try {
                unSuperDisc.setOptions(settings);
                unSuperDisc.setInputFormat(inputDataSet);
                filteredDataSet = Filter.useFilter(inputDataSet, unSuperDisc);
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            numberOfSettings = 1;
            settings = new String[numberOfSettings*2];

            String s1 = "-R";
            String v1 = "first-last";
            //System.out.print("-R : ");
            //v1 = sc.nextLine();
            
            settings[0] = s1;
            settings[1] = v1;
            
            superDisc = new weka.filters.supervised.attribute.Discretize();
            try {
                superDisc.setOptions(settings);
                superDisc.setInputFormat(inputDataSet);
                filteredDataSet = Filter.useFilter(inputDataSet, superDisc);
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    public Instance discretizeInstance(Instance inst) {
        System.out.println();
        System.out.print("Supervised (1) or unsupervised (2) discretize : ");
        int numberOfSettings;
        String[] settings;
        if(sc.nextInt()==2) {
            numberOfSettings = 3;
            settings = new String[numberOfSettings*2];

            String s1 = "-R", 
            s2 = "-B", 
            s3 = "-M";
            String v1 = "first-last";
            String v2 = "10";
            String v3 = "-1";
            //System.out.print("-R : ");
            //v1 = sc.nextLine();
            //System.out.print("-B : ");
            //v2 = sc.nextLine();
            //System.out.print("-M : ");
            //v3 = sc.nextLine();

            settings[0] = s1;
            settings[1] = v1;
            settings[2] = s2;
            settings[3] = v2;
            settings[4] = s3;
            settings[5] = v3;
            
            /*
            weka.filters.unsupervised.attribute.Discretize unsuper = new weka.filters.unsupervised.attribute.Discretize();
            inst.setDataset(inputDataSet);
            try {
                unsuper.setInputFormat(inputDataSet);
                if(unsuper.input(inst)) {
                    inst = unsuper.output();
                }
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }
            */
            unSuperDisc = new weka.filters.unsupervised.attribute.Discretize();
            try {
                unSuperDisc.setOptions(settings);
                unSuperDisc.setInputFormat(inputDataSet);
                Filter.useFilter(inputDataSet, unSuperDisc);
                
                if(unSuperDisc.input(inst)) {
                    inst = unSuperDisc.output();
                }
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            numberOfSettings = 1;
            settings = new String[numberOfSettings*2];

            String s1 = "-R";
            String v1 = "first-last";
            //System.out.print("-R : ");
            //v1 = sc.nextLine();

            settings[0] = s1;
            settings[1] = v1;
            
            /*
            weka.filters.supervised.attribute.Discretize sd = new weka.filters.supervised.attribute.Discretize();
            inst.setDataset(inputDataSet);
            try {
                sd.setInputFormat(inputDataSet);
                if(sd.input(inst)) {
                    inst = sd.output();
                }
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }            
            */
            
            superDisc = new weka.filters.supervised.attribute.Discretize();
            try {
                superDisc.setOptions(settings);
                superDisc.setInputFormat(inputDataSet);
                Filter.useFilter(inputDataSet, superDisc);
                
                if(superDisc.input(inst)) {
                    inst = superDisc.output();
                }
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return inst;
    }
    
    public void train() {
        try {
            System.out.print("Classifier? 1 for NaiveBayes, 2 for MLP, 3 for J48 : "); 
            cType = sc.nextInt();
            if(cType==1) {
                classifier = new NaiveBayes();
            } else if (cType==2) {
                classifier = new MultilayerPerceptron();
            } else {
                classifier = new J48();
            }
            classifier.buildClassifier(filteredDataSet);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void tenFoldEvaluation () {
        try {
            Evaluation eval = new Evaluation(filteredDataSet);
            if (cType==1) {
                NaiveBayes nb = new NaiveBayes();
                eval.crossValidateModel(nb, filteredDataSet, 10, new Random(1));
            } else if (cType==2) {
                MultilayerPerceptron mlp = new MultilayerPerceptron();
                eval.crossValidateModel(mlp, filteredDataSet, 10, new Random(1));
            } else {
                J48 tree = new J48();
                eval.crossValidateModel(tree, filteredDataSet, 10, new Random(1));
            }
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void fullTrainingEvaluation () {
        try {
            Instances test = filteredDataSet;
            
            Evaluation eval = new Evaluation(filteredDataSet);
            eval.evaluateModel(classifier, test);
            
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    } 
    
    public void saveModels(String path) {
        try {
            SerializationHelper.write(path, classifier);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void loadModels(String path) {
        try {
            classifier = (Classifier) weka.core.SerializationHelper.read(path);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public Instance askInstancesFromUser() {
        int nAttributes = inputDataSet.numAttributes();
        int classIndex = nAttributes - 1;
        Instance inst = new DenseInstance(nAttributes);
        String in;
        
        for (int i = 0; i<classIndex; i++) {
            int j = i + 1;
            System.out.print("Attribute (" + j + ") value : ");
            Attribute att = inputDataSet.attribute(i);
            float val = sc.nextFloat();
            inst.setValue(att, val);
        }
        return inst;
    }
    
    public void classifyInstance(Instance inst) {
        try {
            double kelas = classifier.classifyInstance(inst);
            inst.setClassValue(kelas);
            System.out.println("Class prediction : " + inst.stringValue(inst.numAttributes()-1));
            System.out.println();
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void main(String[] args) throws Exception {
        TucilWeka test = new TucilWeka();
        
        try {
            Scanner sc = new Scanner(System.in);
            
            //Header
            System.out.println("-------------------------");
            System.out.println("Welcome to WKWKWKWKWKWKWK");
            System.out.println("Created by : ");
            System.out.println("Geraldi Dzakwan  13514065");
            System.out.println("M. Reza Ramadhan 13514107");
            System.out.println("-------------------------");
            
            //Model
            System.out.println();
            System.out.print("Would you like to train(1) or load(2) model : ");
            
            if ("1".equals(sc.nextLine())) {
                //Load file
                System.out.println();
                System.out.print("Specify dataset file name to load (.arff) : ");
                test.loadData(sc.nextLine());
                
                //Discretize dataset
                System.out.print("Would you like to discretize dataset (y/n) : ");
                if ("y".equals(sc.nextLine())) {
                    System.out.println();
                    test.discretizeData();
                    System.out.println("Data is now discretized");
                }
                
                //Train data
                System.out.println();
                System.out.println("Training data...");
                test.train();
                System.out.println("Model created...");
                
                //Save model 
                System.out.println();
                System.out.print("Would you like to save model (y/n) : ");
                if("y".equals(sc.nextLine())) {
                    System.out.println();
                    System.out.print("Specify model file name (.model) to save : ");
                    test.saveModels(sc.nextLine());
                }
                
                //Evaluating model
                System.out.println();
                System.out.print("How would you like to evaluate model, full(1) 10-fold(2) : ");
                if("1".equals(sc.nextLine())) {
                    System.out.println();
                    System.out.println("Doing full training evaluation : ");
                    test.fullTrainingEvaluation();
                } else {
                    System.out.println();
                    System.out.println("Doing tenFoldEvaluation : ");
                    test.tenFoldEvaluation();
                }
                
                //Classification
                String in = "";
                do {
                    System.out.println();
                    System.out.print("Would you like to classify new instance (y/n) : ");
                    in = sc.nextLine();
                    if("y".equals(in)) {
                        Instance i = test.askInstancesFromUser();
                        Instance temp = i;
                        System.out.println();
                        System.out.print("Is your dataset discretized before? (y/n) : ");
                        if("y".equals(sc.nextLine())) {
                            i = test.discretizeInstance(i);
                        }

                        System.out.println();
                        i.setDataset(test.inputDataSet);
                        test.classifyInstance(i);

                        //Save new instance
                        temp.setDataset(test.inputDataSet);
                        temp.setClassValue(i.stringValue(i.numAttributes()-1));
                        System.out.print("Would you like to add this new instance to the dataset? (y/n) : ");
                        Scanner sc2 = new Scanner(System.in);
                        String in2 = sc2.nextLine();
                        if("y".equals(in2)) {
                            test.inputDataSet.add(temp);
                            System.out.println("Instances added");
                            System.out.println();
                            System.out.print("Would you like to save this new dataset to arff file (y/n) : ");
                            String in3 = sc2.nextLine();
                            if ("y".equals(in3)) {
                                System.out.println();
                                System.out.print("Specify dataset file name (.arff) to save : ");
                                test.saveData(sc2.nextLine());
                                System.out.println("File saved");
                            }
                        }
                    }
                } while ("y".equals(in));
            } else {
                //Load file
                System.out.println();
                System.out.print("Specify dataset file name to load (.arff) : ");
                String inputDataSet = sc.nextLine();
                test.loadData(inputDataSet);
                
                //Load model 
                System.out.println();
                System.out.print("Specify model file name to load (.model) : ");
                test.loadModels(sc.nextLine());
                
                //Classification
                String in = "";
                do {
                    System.out.println();
                    System.out.print("Would you like to classify new instance (y/n) : ");
                    in = sc.nextLine();
                    if("y".equals(in)) {
                        Instance i = test.askInstancesFromUser();
                        Instance temp = i;
                        System.out.println();
                        System.out.print("Is your dataset discretized before? (y/n) : ");
                        if("y".equals(sc.nextLine())) {
                            i = test.discretizeInstance(i);
                        }

                        System.out.println();
                        i.setDataset(test.inputDataSet);
                        test.classifyInstance(i);

                        //Save new instance
                        temp.setDataset(test.inputDataSet);
                        temp.setClassValue(i.stringValue(i.numAttributes()-1));
                        System.out.print("Would you like to add this new instance to the dataset? (y/n) : ");
                        Scanner sc2 = new Scanner(System.in);
                        String in2 = sc2.nextLine();
                        if("y".equals(in2)) {
                            test.inputDataSet.add(temp);
                            System.out.println("Instances added");
                            System.out.println();
                            System.out.print("Would you like to save this new dataset to arff file (y/n) : ");
                            String in3 = sc2.nextLine();
                            if ("y".equals(in3)) {
                                System.out.println();
                                System.out.print("Specify dataset file name (.arff) to save : ");
                                test.saveData(sc2.nextLine());
                                System.out.println("File saved");
                            }
                        }
                    }
                } while ("y".equals(in));
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
