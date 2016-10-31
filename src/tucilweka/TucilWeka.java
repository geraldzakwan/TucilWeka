/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Geraldi Dzakwan
 */
public class TucilWeka {

    public void loadAndSaveData() throws FileNotFoundException {
        String loadFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/iris.arff";
        try {
            DataSource ds = new DataSource(loadFilePath);
            Instances newDataSet = ds.getDataSet();
            //Ini kalo instances langsung
            //Instances newDataSet = new Instances(new BufferedReader(new FileReader(loadFilePath)));            

            System.out.println(newDataSet.toSummaryString());

            ArffSaver aSaver = new ArffSaver();
            aSaver.setInstances(newDataSet);
            String saveFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/saved.arff";
            aSaver.setFile(new File(saveFilePath));
            aSaver.writeBatch();
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
            test.loadAndSaveData();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
