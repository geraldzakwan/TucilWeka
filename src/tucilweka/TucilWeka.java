package tucilweka;

//Import untuk std I/O, FIle I/O, Exception Logging, dan Random
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Random;

//Impport untuk fungsi Weka secara umum seperti
//instances, save file arff, dan data source
import weka.core.*;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

//Import untuk preprocessor filter
import weka.filters.Filter;

//Import untuk classifier secara umum dan dua jenis classifier yakni J48 dan MLP
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.MultilayerPerceptron;

//Import untuk evaluasi model pembelajaran
import weka.classifiers.Evaluation;
/**
 *
 * @author Geraldi Dzakwan
 * @author M. Reza Ramadhan
 */
public class TucilWeka {
    
    DataSource dataSource;
    Instances inputDataSet;
    Instances outputDataSet;
    Classifier j48Classifier;
    Classifier MLP;
    int cType;
    Scanner sc;
    
    weka.filters.unsupervised.attribute.Discretize unSuperDisc;
    weka.filters.supervised.attribute.Discretize superDisc;
    
    public TucilWeka() {
        j48Classifier = new J48();
        MLP = new MultilayerPerceptron();
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
        aSaver.setInstances(outputDataSet);
//        String saveFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/saved.arff";
        //String saveFilePath = "saved.arff";
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
        
        unSuperDisc = new weka.filters.unsupervised.attribute.Discretize();
        try {
            unSuperDisc.setOptions(settings);
            unSuperDisc.setInputFormat(inputDataSet);
            outputDataSet = Filter.useFilter(inputDataSet, unSuperDisc);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /*
        Function to train the classifiers
    */
    public void train() {
        try {
            System.out.print("Classifier? 1 for j48, 2 for MLP : "); 
            cType = sc.nextInt();
            if(sc.nextInt()==1) {
                j48Classifier.buildClassifier(outputDataSet);
            } else {
                MLP.buildClassifier(outputDataSet);
            }
            
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void tenFoldEvaluation () {
        try {
            Evaluation eval = new Evaluation(outputDataSet);
            if (cType==1) {
                J48 tree = new J48();
                eval.crossValidateModel(tree, outputDataSet, 10, new Random(1));
            } else {
                MultilayerPerceptron mlp = new MultilayerPerceptron();
                eval.crossValidateModel(mlp, outputDataSet, 10, new Random(1));
            }
            System.out.println(eval.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void fullTrainingEvaluation () {
        try {
            Instances test = outputDataSet;
            
            Evaluation eval = new Evaluation(outputDataSet);
            
            if(cType==1) {
                eval.evaluateModel(j48Classifier, test);
            } else {
                eval.evaluateModel(MLP, test);
            }
            
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
            Scanner sc = new Scanner(System.in);
            System.out.print("File to load : ");
            test.loadData(sc.nextLine());

            test.discretizeData();
            
            test.train();
            
            //pilih either fulltraining atau 10-fold
            
            //buat hipotesis
            
            //tampilin hipotesis
            
            //simpan hipotesis
            
            //baca instans baru
            
            //klasifikasi
            System.out.print("Model to save : ");
            test.saveModels(sc.nextLine());
            
            System.out.print("1 for full or 2 for tenFold : ");
            if(sc.nextInt()==1) {
                System.out.println("Doing full training evaluation : ");
                test.fullTrainingEvaluation();
            } else {
                System.out.println("Doing tenFoldEvaluation : ");
                test.tenFoldEvaluation();
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
