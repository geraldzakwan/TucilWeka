package tucilweka;

//Import untuk std I/O, FIle I/O, Exception Logging, dan Random
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
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
//Kelas Java yang diimplementasikan
public class TucilWeka {
    
    //Untuk mengambil dataset dari file eksternal
    DataSource dataSource;
    //Input dataset
    Instances inputDataSet;
    //Dataset yang sudah di-filter
    Instances filteredDataSet;
    //Dataset unclassified
    Instances unsetData;
    //Nanti ini diapus
    Classifier j48Classifier;
    Classifier MLP;
    
    //Variabel classifier
    Classifier classifier;
    //Menyatakan classifier jenis apa 
    int cType;
    
    //Stdin
    Scanner sc;
    
    //Filter yang digunakan yakni supervised & unsupervised discretize
    weka.filters.unsupervised.attribute.Discretize unSuperDisc;
    weka.filters.supervised.attribute.Discretize superDisc;
    
    //Konstruktor
    public TucilWeka() {
        //Inisialisasi scanner
        //j48Classifier = new J48();
        //MLP = new MultilayerPerceptron();
        sc = new Scanner(System.in);
    }
    
    //Load unfiltered dataset
    public void loadData(String loadFilePath) throws FileNotFoundException {
        try {
            //Mengambil data source dari file arff
            dataSource = new DataSource(loadFilePath);
            //Mengambil dataset dari data source
            inputDataSet = dataSource.getDataSet();
            //Mengeset bahwa atribut kelas/klasifikasi ada di atribut paling akhir
            inputDataSet.setClassIndex(inputDataSet.numAttributes() - 1);
            //Output info dari dataset
            System.out.println(inputDataSet.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    //Save filtered dataset
    public void saveData(String saveFilePath) {
        //Membuat variabel saver
        ArffSaver aSaver = new ArffSaver();
        //Mengeset instance menjadi filtered dataset
        aSaver.setInstances(filteredDataSet);
        try {
            //Mengeset filepath dan namafile
            aSaver.setFile(new File(saveFilePath));
            //Write ke disk
            aSaver.writeBatch();
        } catch (IOException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    //Melakukan filter dataset menggunakan discretize
    public Instances discretizeData(Instances arrInst) throws Exception  {
        //Menentukan filter apa yang digunakan (supervised/unsupervised)
        Instances retVal = dataSource.getDataSet();
        System.out.println("Supervised (1) atau unsupervised (2) discretize : ");
        int numberOfSettings;
        String[] settings;
        if(sc.nextInt()==2) {
            numberOfSettings = 3;
            settings = new String[numberOfSettings*2];

            String s1 = "-R", 
            s2 = "-B", 
            s3 = "-M";
            String v1 = "", v2 = "", v3 = "";

            System.out.print("-R : ");
            //v1 = "first-last";
            v1 = sc.nextLine();
      
            System.out.print("-B : ");
            //v2 = "10";
            v2 = sc.nextLine();
            
            System.out.print("-M : ");
            //v3 = "-1";
            v3 = sc.nextLine();

            settings[0] = s1;
            settings[1] = v1;
            settings[2] = s2;
            settings[3] = v2;
            settings[4] = s3;
            settings[5] = v3;
            
            unSuperDisc = new weka.filters.unsupervised.attribute.Discretize();
            try {
                unSuperDisc.setOptions(settings);
                unSuperDisc.setInputFormat(arrInst);
                retVal = Filter.useFilter(arrInst, unSuperDisc);
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            numberOfSettings = 1;
            settings = new String[numberOfSettings*2];

            String s1 = "-R";
            //v1 = first-last
            String v1 = "";

            System.out.print("-R : ");
            //v1 = "first-last";
            v1 = sc.nextLine();

            settings[0] = s1;
            settings[1] = v1;
            
            superDisc = new weka.filters.supervised.attribute.Discretize();
            try {
                superDisc.setOptions(settings);
                superDisc.setInputFormat(arrInst);
                retVal = Filter.useFilter(arrInst, superDisc);
            } catch (Exception ex) {
                Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return retVal;
    }
    
    /*
        Function to train the classifiers
    */
    public void train() {
        try {
            System.out.print("Classifier? 1 for j48, 2 for MLP : "); 
            cType = sc.nextInt();
            if(cType==1) {
                classifier = new J48();
                //j48Classifier.buildClassifier(filteredDataSet);
                classifier.buildClassifier(filteredDataSet);
            } else {
                classifier = new MultilayerPerceptron();
                //MLP.buildClassifier(filteredDataSet);
                classifier.buildClassifier(filteredDataSet);
            }
            
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void tenFoldEvaluation () {
        try {
            Evaluation eval = new Evaluation(filteredDataSet);
            if (cType==1) {
                J48 tree = new J48();
                eval.crossValidateModel(tree, filteredDataSet, 10, new Random(1));
            } else {
                MultilayerPerceptron mlp = new MultilayerPerceptron();
                eval.crossValidateModel(mlp, filteredDataSet, 10, new Random(1));
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
            
            /*
            if(cType==1) {
                eval.evaluateModel(j48Classifier, test);
            } else {
                eval.evaluateModel(MLP, test);
            }
            */
            
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
            //SerializationHelper.write(path, j48Classifier);
            SerializationHelper.write(path, classifier);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void loadModels(String path) {
        try {
            //j48Classifier = (Classifier) weka.core.SerializationHelper.read(path);
            classifier = (Classifier) weka.core.SerializationHelper.read(path);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void classifyInstance(Instance inst) {
        try {
            //double classLabel = j48Classifier.classifyInstance(inst);
            ArrayList<Attribute> atts = new ArrayList<Attribute>();
            for (int i = 0; i < inputDataSet.numAttributes(); i++) {
                atts.add(inputDataSet.attribute(i).copy(inputDataSet.attribute(i).name()));
            }
            System.out.println("Oi : " + atts.size());
            
            unsetData = new Instances("TestInstances", atts, 0);
            unsetData.setClassIndex(inputDataSet.numAttributes() - 1);
            
            inst.setClassValue("Iris-virginica");
            unsetData.add(inst);
            
            unsetData = discretizeData(unsetData);
            unsetData.setClassIndex(inputDataSet.numAttributes() - 1);
            
            inst.setDataset(unsetData);
            //System.out.println(classifier);
            
            double classLabel = classifier.classifyInstance(unsetData.firstInstance());
            inst.setClassValue(classLabel);
            System.out.println(inst.numAttributes());
            
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

            test.filteredDataSet = test.discretizeData(test.inputDataSet);
            
            test.train();
            
            Instance i = test.askInstancesFromUser();
            test.classifyInstance(i);
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
