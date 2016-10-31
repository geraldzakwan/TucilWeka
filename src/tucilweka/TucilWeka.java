/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.Scanner;

/**
 *
 * @author Geraldi Dzakwan
 */
public class TucilWeka {
    
    DataSource dataSource;
    Instances inputDataSet;
    Instances outputDataSet;

    public void loadData() throws FileNotFoundException {
        String loadFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/iris.arff";
        try {
            dataSource = new DataSource(loadFilePath);
            inputDataSet = dataSource.getDataSet();
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
        String saveFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/saved.arff";
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
        Scanner sc = new Scanner(System.in);
        
        System.out.print("-R : ");
        v1 = sc.nextLine();
        
        System.out.print("-B : ");
        v2 = sc.nextLine();
        
        System.out.print("-M : ");
        v3 = sc.nextLine();
        
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
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        System.out.println("import working");
        TucilWeka test = new TucilWeka();
        try {
            test.loadData();
            test.discretizeData();
            test.saveData();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
