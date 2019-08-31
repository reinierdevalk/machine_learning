package machineLearning;

import java.util.List;

public class RelativeTrainingExample {

	private List<Double> betterVal;
	private List<Double> worseVal;	


	public RelativeTrainingExample(List<Double> better, List<Double> worse) {
		setBetterVal(better);
		setWorseVal(worse);
	}


	public List<Double> getBetterVal() {
		return betterVal;
	}


	public List<Double> getWorseVal() {
		return worseVal;
	}


	private void setBetterVal(List<Double> better) {
		betterVal = better;
	}


	private void setWorseVal(List<Double> worse) {
		worseVal = worse;
	}
	
}
