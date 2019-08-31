/*
 * File TestFuzzyProgram.java
 * Created on 28.05.2004
 */
package machineLearning;

import java.util.*;

import junit.framework.TestCase;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

import antlr.SemanticException;
import base.*;
import de.uos.fmt.musitech.utility.general.WrongArgumentException;
import fuzzy.clauses.*;
import fuzzy.operators.SigmoidOperator;
import fuzzy.training.TrainingNode;
import fuzzy.training.algorithms.BackProp;

/**
 * Test Class for clauses
 * 
 * @author Tillman Weyde & Reinier de Valk
 * 
 */
public class TestClauses2 extends TestCase {

	D d1 = new D(1, -2, 3, 4);
	G g1 = new G();
	FactClause fcl = new FactClause();
	SimpleClause scl;

	

	
	/**
	 * This test runs learning by directly modifying the weights 
	 * of the network and by using the BackProp algorithm for 
	 * comparison. TODO automatic comparison not yet implemented.
	 */
	public void testSimpleClauseTraining() {
	  // ENCOG:
		// Make features
		List<Double> feature = new ArrayList<Double>();
		feature.add(1.0);
		feature.add(-2.0);
		List<List<Double>> features = new ArrayList<List<Double>>();
		features.add(feature);
		
		// Make labels
		List<Double> label = new ArrayList<Double>();
		label.add(1.0);
		List<List<Double>> labels = new ArrayList<List<Double>>();
		labels.add(label);
		
		// Make chordLabels
		List<List<List<Double>>> chordLabels = new ArrayList<List<List<Double>>>();
		List<List<Double>> labelsChord1 = new ArrayList<List<Double>>();
		labelsChord1.add(label);
		chordLabels.add(labelsChord1);
		
		// Create NN with given weights
		BasicNetwork network = new BasicNetwork() ;
		network.addLayer (new BasicLayer (null, false, 2)); // input
		network.addLayer (new BasicLayer (new ActivationSigmoid(), false, 1)); // output
		network.getStructure().finalizeStructure();
		network.reset();
		network.setWeight(0, 0, 0, -0.5);
		network.setWeight(0, 1, 0, 0.25);
		
		EncogNeuralNetworkManager netMan = new EncogNeuralNetworkManager(network); 
//		List<Double> weights = new ArrayList<Double>();
//		weights.add(-0.5);
//		weights.add(0.25);
//		netMan.initWeightsFromList(weights);
		
		double[] outputInitial, outputNew, outputOld; // vars for storing output; all arrays contain only a single element
		outputInitial = netMan.evalNetwork(features.get(0));
		outputOld = outputInitial; 
		System.out.println("outputInitial = " + outputInitial[0]);
		assertEquals(1 / (1 + Math.E), outputInitial[0], .0001);// value that we have
		
		// Train the NN 10 times
		netMan.createTrainingExamplesNoteToNote(features, labels, false, false);
		double eta = 0.5;
		double lambda = 0.0;
		int cycles = 1;
		Double[] outputsOfEncog = new Double[10];
		for (int i = 0; i < 10; i++) { 
//			netMan.trainNetwork(cycles, eta, lambda, false, true);
			outputNew = netMan.evalNetwork(features.get(0));
			outputsOfEncog[i] = outputNew[0];
			assertTrue("output_new should be greater than output_old", outputNew[0] > outputOld[0]);
			outputOld = outputNew; 
		}
		String resultsEncog = "Outputs of Encog:   ";
		for (int i = 0; i < outputsOfEncog.length; i++) {
		  resultsEncog += outputsOfEncog[i] + ", ";
		}
		System.out.println(resultsEncog);
	
	  // TILLMAN:
		// Build a simple network
		D d1 = new D(1, -2, 3, 4);
		List<Clause> clauseList = new ArrayList<Clause>();
		FactClause fcl1 = new FactClause();
		FactClause fcl2 = new FactClause();
		try {
			fcl1.setName("", "base.D.get1");
			fcl2.setName("", "base.D.get2");
		} catch (SemanticException e1) {
			fail(e1.toString());
		} catch (RuntimeException e1) {
			fail(e1.toString());
		}
		clauseList.add(fcl1);
		clauseList.add(fcl2);
		scl = new SimpleClause(clauseList, new SigmoidOperator());
  
		double output, output_new, output_old;
		Double[] outputsOfTillman = new Double[10];
		try {
			// Create the training data structures
			Map<Clause, TrainingNode> cn_map = new HashMap<Clause, TrainingNode>();
			TrainingNode tn;
			tn = scl.createTrainingNodes(d1, cn_map);

			// set the weights and calculate the network output
			scl.setWeight(0, -.5);
			scl.setWeight(1, .25);
			output = scl.calcValueTraining(tn);
			output_old = output;
			System.out.println("output_old = " + output_old);
			assertEquals(1 / (1 + Math.E), output, .0001);

			BackProp algo = new BackProp();
			algo.setCycles(1);
			algo.setNu(0);
			algo.setLrate(eta);
			scl.setMinWeight(-1000);
			scl.setMaxWeight(1000);
      			
			for (int i = 0; i < 10; i++) {
				algo.resetWDeltaChange(scl);
				scl.resetWeightDelta();
				scl.calcDeltas(tn, 1);
				algo.applyWeightDelta(scl);
				output_new = scl.calcValueTraining(tn);
				outputsOfTillman[i] = output_new;
				assertTrue("output_new should be greater than output_old", output_new > output_old);
				output_old = output_new;
			// store the output value for each iteration
			}
		} catch (WrongArgumentException e) {
			fail(e.toString());
			return;
		}
		// printing for visual checking
		String resultsTillman = "Outputs of Tillman: ";
		for (int i = 0; i < outputsOfTillman.length; i++) {
			resultsTillman += outputsOfTillman[i] + ", ";
		}
		// jUnit testing
		System.out.println(resultsTillman);
		for (int i = 0; i < outputsOfTillman.length; i++) {
			assertEquals(outputsOfEncog[i], outputsOfTillman[i]);
		}
		
		return;
	}

}
