# TimeMixer-KAN
TimeMixer-KAN is an innovative model that combines the TimeMixer architecture with Kolmogorov-Arnold Networks (KAN) to enhance time series forecasting, particularly for stock trading. This model, even without fine-tuning, has demonstrated superior performance over the standard TimeMixer model in short-term stock price forecasting.

## Overview
TimeMixer-KAN is designed to leverage the strengths of both TimeMixer and Kolmogorov-Arnold Networks to process sequential data. By integrating these models, TimeMixer-KAN can effectively handle complex patterns in time series data, making it particularly useful for forecasting tasks in financial markets.

## Key Features
Hybrid Architecture: Combines TimeMixer’s sequential modeling capabilities with the Kolmogorov-Arnold Networks for increased flexibility and performance.

Short-Term Forecasting: Outperforms the standalone TimeMixer model, achieving a validation MSE loss of 0.0012793 on the stock 'AU8U.SI' (not yet fine-tuned).

## Usage
### Training
To train the model, use the following command:
```bash
python ./stocks.py --tmkan --train --save_path ./timemixer_kan/
```
### Resume Training
To resume training, 
```bash
python ./stocks.py --tmkan --train --resume --save_path ./timemixer_kan/
```

### Testing
To test the model, use:
```bash
python ./stocks.py --tmkan --test --save_path ./timemixer_kan/
```

## Contributing
This is a small experiment project, and any feedback or suggestions are greatly appreciated! Feel free to comment, open issues, or submit pull requests to help improve the model and its capabilities. I am using both the TimeMixer model concept (https://github.com/kwuking/TimeMixer) and KAN (https://github.com/pg2455/KAN-Tutorial)

## Things are not done
- Fine tuning the model including tensorboard and predicted visualization
- Experiment on long and short term forecasting
- More trading stocks for comparison purposes

## Further Improvements
- MoE haven't done yet
- Add LLM model like Gemini, Llama, etc
- Data Augmentation

