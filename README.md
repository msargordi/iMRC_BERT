# iMRC_BERT

This repository contains the implementation of a BERT-based model for machine reading comprehension. Below you will find instructions on how to set up and run examples.

## How to Run Examples

To run an example with the BERT model, you will need a configuration file in YAML format. Use the following command from the command line:

```bash
python main.py <config>.yaml
```
Replace `<config>` with the name of your configuration file.

## System Architecture

Below is an overview of the system architecture of our agent.

### Overview of Agent
![Agent Overview](/utils/image.png)

The system consists of:
- **Frozen LLM**: A pre-trained BERT/RoBERTa model that is utilized without further training during operations.
- **FC Layers**: Fully connected layers.
- **Output Actions**: The agent generates actions `a_t` based on the processed inputs. `a_t` could be: search with a token, go to next/previous sentence/chunk, or end. 

This conceptual overview provides a high-level understanding of the interconnections and the operational logic of the agent.
