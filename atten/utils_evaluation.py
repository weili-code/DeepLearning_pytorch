# imports from installed libraries
import os
import numpy as np
import random
import torch
from torch import nn
from distutils.version import LooseVersion as Version
import sacrebleu
from tqdm import tqdm


def set_all_seeds(seed):
    """
    Set the seed for all relevant RNGs to ensure reproducibility across runs.

    This function sets a fixed seed for random number generators in os, random,
    numpy, and torch, ensuring that the same sequences of random numbers will be
    generated across different program executions when the same seed is used. It is
    particularly useful when trying to reproduce results in machine learning experiments.

    credits: Sebastian Raschka
    """

    # Set the seed for generating random numbers in Python's os module
    os.environ["PL_GLOBAL_SEED"] = str(seed)

    # Set the seed for the default Python RNG
    random.seed(seed)

    # Set the seed for numpy's RNG
    np.random.seed(seed)

    # Set the seed for PyTorch's RNG
    torch.manual_seed(seed)

    # Ensure that CUDA kernels' randomness is also seeded if available
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """
    Enforces deterministic behavior in PyTorch operations to ensure reproducibility.

    This function configures PyTorch to behave deterministically, especially when running
    on a CUDA (GPU) environment. It disables certain optimizations that introduce non-determinism,
    making sure that the same inputs across different runs produce the same outputs.

    Note: Some PyTorch operations do not support deterministic mode, and using this function
    may have performance implications due to disabled optimizations.

    credits: Sebastian Raschka
    """

    # If CUDA (GPU support) is available, set related options for deterministic behavior
    if torch.cuda.is_available():
        # Disable the auto-tuner that finds the best algorithm for a specific input configuration.
        # This is necessary for reproducibility as different algorithms might produce slightly different results.
        torch.backends.cudnn.benchmark = False

        # Enable CuDNN deterministic mode. This ensures that convolution operations are deterministic.
        torch.backends.cudnn.deterministic = True

    # Set the deterministic flag based on the version of PyTorch.
    # Different versions of PyTorch use different functions to enforce deterministic algorithms.
    if torch.__version__ <= Version("1.7"):
        # For versions 1.7 or older, use `torch.set_deterministic`
        torch.set_deterministic(True)
    else:
        # From version 1.8 forward, use `torch.use_deterministic_algorithms`
        torch.use_deterministic_algorithms(True)


def generate_tgt_mask(tgt, pad_idx):
    """
    Generates a target mask to hide padding tokens and prevent attention to future tokens,
    combinng padding mask and look-ahead mask.

    Args:
        tgt (Tensor): Target input tensor of shape (batch_size, max_seq_length). max_seq_lenth = max seq length in the batch
        pad_idx (int): Index used for padding tokens.

    Returns:
        Tensor: Combined mask tensor of shape (batch_size, 1, max_seq_length, max_seq_length).
    """
    # Determine the sequence length from the target tensor
    seq_len = tgt.size(1)  # seq_len is the second dimension of tgt

    # Create a lower triangular matrix of ones (including diagonal)
    # This 'no_future_mask' is used to prevent attention to future tokens.
    # It's a square matrix of shape (seq_len, seq_len).
    # In this matrix, each row corresponds to a token, and the columns it can attend to.
    # For example, the first row (first token) can only attend to the first column (itself),
    # the second row can attend to the first and second columns, and so on.
    no_future_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()

    # Create a padding mask by comparing each element in the target tensor with the padding index
    # This results in a boolean tensor where 'True' indicates a non-padding token
    # and 'False' indicates a padding token.
    # The mask is then reshaped to (batch_size, 1, 1, seq_len) to be compatible with the shape expected
    # by the attention mechanism.
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    # Combine the no_future_mask and the pad_mask using a logical AND operation
    # This ensures that the model does not attend to future tokens or padding tokens.
    # The resulting combined mask is of shape (batch_size, 1, seq_len, seq_len), suitable
    # for the attention mechanism in the transformer model.
    combined_mask = pad_mask & no_future_mask

    return combined_mask


def generate_src_mask(src, pad_idx):
    """
    Generates a source mask to hide padding tokens in the input tensor.

    Args:
        src (Tensor): Source input tensor of shape (batch_size, max_seq_length).
        pad_idx (int): Index used for padding tokens.

    Returns:
        Tensor: Mask tensor of shape (batch_size, 1, 1, max_seq_length).
    """
    # Compare each element in the source tensor with the padding index.
    # The result is a boolean tensor where 'True' indicates a non-padding token
    # and 'False' indicates a padding token.
    # src shape: (batch_size, max_seq_length)
    mask = src != pad_idx

    # Add two singleton dimensions using unsqueeze.
    # The first unsqueeze adds a singleton dimension at position 1,
    # resulting in shape (batch_size, 1, max_seq_length).
    # The second unsqueeze adds a singleton dimension at position 2,
    # resulting in the final mask shape (batch_size, 1, 1, max_seq_length).
    mask = mask.unsqueeze(1).unsqueeze(2)

    return mask


def calculate_bleu(tgt_output, output, de_vocab):
    """
    Calculates the BLEU score for the target output and the model's output.

    Args:
    tgt_output (Tensor): The ground truth target output tensor (indices).
                         (batch_size, max_tgt_output_seq_len - 1), which exludes the first token <sos>.
    output (Tensor): The predicted output tensor (indices) from the model, shape (batch_size, max_tgt_output_seq_len - 1).
                     Usually, output is obtained as model_output.argmax(-1)
                     where model_output is the return from forwar method of model <Transformer>;
                     model_output shape (batch_size, max_tgt_input_seq_len - 1, tgt_vocab_size);
                     so output is prediction at tgt_input (batch_size, max_tgt_input_seq_len - 1);
                     i.e., predictions at the first token (after <sos>), and all remaining tokens (including possibly <pad>) excluding prediction for <eos>.
    de_vocab:

    Returns:
        float: The computed BLEU score.
    """

    # Convert tensors to numpy arrays for easier manipulation
    # This is done to facilitate operations that are not as straightforward with PyTorch tensors
    tgt_output = tgt_output.cpu().numpy()
    output = output.cpu().numpy()

    # Initialize lists to store the reference translations and the hypotheses (predictions)
    refs = []  # Will contain the reference sentences
    hyps = []  # Will contain the predicted sentences

    # Iterate over each pair of target and predicted outputs
    for tgt, pred in zip(tgt_output, output):
        # Convert the target sequence of token indices to a string, excluding special tokens
        # such as padding (<pad>), end-of-sequence (<eos>), and start-of-sequence (<sos>)
        ref = " ".join(
            [
                de_vocab.itos[t]
                for t in tgt
                if t not in (de_vocab["<pad>"], de_vocab["<eos>"], de_vocab["<sos>"])
            ]
        )
        # Convert the predicted sequence of token indices to a string, similar to above
        hyp = " ".join(
            [
                de_vocab.itos[t]
                for t in pred
                if t not in (de_vocab["<pad>"], de_vocab["<eos>"], de_vocab["<sos>"])
            ]
        )

        # Append the cleaned reference and hypothesis to their respective lists
        refs.append(ref)
        hyps.append(hyp)

    # Calculate the BLEU score using the sacrebleu library
    # sacrebleu is used for consistent and standard BLEU score calculation
    # 'force=True' is used to force the calculation even if there are issues (like short sentences)
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True).score

    # Return the computed BLEU score
    return bleu


def inference(model, src, en_vocab, de_vocab, device):
    """
    With source sequences src, generate output sequences sequentially up to some maximal time steps or until
    all output sequences reached <eos>. Output at each time is picked as the one with highest probability.

    - model
    - src
    - en_vocab
    - de_vocab
    - device

    """
    # src: shape (batch_size, max_src_seq_len)

    model.eval()  # Set the model to evaluation mode

    # Generate a source mask to ignore padding tokens in source sequences
    src_mask = generate_src_mask(src, en_vocab["<pad>"])

    # Initialize target input tensors with <sos> (start of sequence) tokens
    # Each sequence in the batch starts with an <sos> token
    tgt_input = torch.full(
        (src.size(0), 1), de_vocab["<sos>"], dtype=torch.long, device=device
    )
    # tgt_input shape (batch_size, 1): consists of index of <sos>

    # Create a flag for each sequence in the batch to track if <eos> (end of sequence) is reached
    eos_flags = torch.zeros(src.size(0), dtype=torch.bool, device=device)
    # eos_flags will be (batch_size), where each element corresponds to a sequence
    # in the batch and indicates whether <eos> has been generated for that sequence.

    # Perform inference for each target token without computing gradients
    with torch.no_grad():
        for _ in range(
            70
        ):  # Iterate up to a maximum of 70 steps (or until all sequences reach <eos>)
            # Generate a target mask for the current target input
            tgt_mask = generate_tgt_mask(tgt_input, de_vocab["<pad>"])

            # Forward pass through the model to get the output
            output = model(src, tgt_input, src_mask, tgt_mask)
            # output shape (batch_size, tgt_input.shape[1], tgt_vocab_size)
            # tgt_input.shape[1]: is the length of the target sequences currently
            # being processed (including the <sos> token and any tokens generated so far).

            # Get the most probable next token from the output
            # output.argmax(2): Get the index of the max log-probability (logits) for each position
            # [:, -1]: Take the last token from each sequence in the batch
            # .unsqueeze(1): Reshape to add a sequence length dimension
            next_tokens = output.argmax(2)[:, -1].unsqueeze(1)
            # next_tokens: [batch_size, 1]. each sequence in the batch has a
            # single new token in a separate row.

            # Concatenate the new tokens to the existing target input sequence
            tgt_input = torch.cat((tgt_input, next_tokens), dim=1)

            # Update the eos_flags for sequences that have generated <eos>
            # |= is a bitwise OR assignment operator, updating the flag if <eos> is generated
            eos_flags |= next_tokens.squeeze() == de_vocab["<eos>"]
            # RHS: shape [batch_size] boolean tensor
            # This updates each element in eos_flags to True if it was already
            # True or if the corresponding element in (next_tokens.squeeze() == DE_VOCAB['<eos>'])
            #  is True

            # Break the loop if all sequences in the batch have generated <eos> or reached maximum length
            if torch.all(eos_flags):
                break

            # note that in this loop, we allow some sequence with <eos> to continue to be generated

    # Convert the target input tensors to translated sentences
    translated_sentences = []
    for i in range(tgt_input.size(0)):  # Iterate over each sequence in the batch
        translated_tokens = []
        for token in tgt_input[i][1:]:  # Skip the first token (<sos>)
            if token == de_vocab["<eos>"]:
                break  # Stop at the first <eos> token
            else:
                translated_tokens.append(
                    de_vocab.itos[token.item()]
                )  # Convert token index to string

        # Join the tokens to form the translated sentence
        translated_sentence = " ".join(translated_tokens)
        translated_sentences.append(translated_sentence)

    return translated_sentences  # a list of translated strings (sentences)


def evaluate_test_set_bleu(model, test_dataloader, en_vocab, de_vocab, device):
    """
    Evaluate the BLEU score of a translation model on a test dataset.

    This function iterates over a given test dataloader, using a specified model
    to translate source sentences from English to German. It compares the model's translations
    against the ground truth German sentences in the test set to calculate the BLEU score.
    The function also outputs an example sentence, its ground truth translation, and the model's
    translation for inspection. The BLEU score is calculated using sacrebleu module.

    https://github.com/mjpost/sacrebleu

    Args:
    - model (nn.Module): The translation model to be evaluated.
    - test_dataloader (DataLoader): A DataLoader containing the test dataset, where each
      batch includes source (English) and target (German) sentences.
    - en_vocab
    - de_vocab
    - device

    Returns:
    - bleu_score (float): The BLEU score for the entire test dataset, indicating the quality
      of the translations.
    - ground_truth_sentences (list of str): List of ground truth sentences in German.
    - translated_sentences (list of str): List of sentences translated by the model.

    The function also prints an example source sentence, its ground truth translation, and the
    corresponding machine translation for qualitative analysis.
    """
    # Lists to store the translated sentences and their corresponding ground truths
    translated_sentences = []
    ground_truth_sentences = []

    # Iterate over all batches in the test dataloader
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        src, tgt_output = batch  # Unpack the batch into source and target outputs
        src, tgt = src.to(device), tgt_output.to(device)

        # Convert target output sequences to actual sentences in German
        # tgt_output is a tensor of shape [batch_size, max_seq_length] with token indices
        # This loop converts each sequence of token indices in tgt_output to a
        # excluding <pad>, <sos> and <eos> sentence (excluding <pad>, <sos> and <eos>)
        tgt_sentences = [
            " ".join(
                [
                    de_vocab.itos[token.item()]
                    for token in sequence
                    if token.item()
                    not in [de_vocab["<pad>"], de_vocab["<sos>"], de_vocab["<eos>"]]
                ]
            )
            for sequence in tgt_output
        ]

        # Use the inference function to translate source sentences
        translations = inference(model, src, en_vocab, de_vocab, device)
        # translations is a list of strings, where each string is a translated sentence

        # Extend the lists with the newly translated sentences and their corresponding ground truths
        translated_sentences.extend(translations)
        # lastly,  len(translated_sentences) = total number of sequences processed across all batches
        ground_truth_sentences.extend(tgt_sentences)

        # WL: I think the original code here is incorrect
        # ground_truth_sentences.extend([[tgt] for tgt in tgt_sentences])

    # Calculate the BLEU score using sacrebleu for the entire corpus of translations
    bleu_score = sacrebleu.corpus_bleu(
        translated_sentences, [ground_truth_sentences]
    )  # see calculate_bleu() function.
    return bleu_score, ground_truth_sentences, translated_sentences


###### ---------------------------------------   ############
######  Playgroud: explain sacrebleu.corpus_bleu ############

# # Example to illustrate ground_truth_sentences.extend([[tgt] for tgt in tgt_sentences])

# # Let's assume we have two batches of ground truth sentences (tgt_sentences) from a test dataset
# batch1_tgt_sentences = ["Der Hund läuft.", "Das Wetter ist schön."]
# batch2_tgt_sentences = ["Das Auto ist neu.", "Das Haus ist groß."]

# # Initialize an empty list for ground_truth_sentences
# ground_truth_sentences = []

# # Extending ground_truth_sentences with the first batch
# ground_truth_sentences.extend([[tgt] for tgt in batch1_tgt_sentences])
# # Extending ground_truth_sentences with the second batch
# ground_truth_sentences.extend([[tgt] for tgt in batch2_tgt_sentences])

# # Print the ground_truth_sentences list
# ground_truth_sentences


# # Translated sentences
# translated_sentences = [
#     "Der Hund rennt.",
#     "Das Wetter ist gut.",
#     "Ein neues Auto.",
#     "Das Haus ist großartig.",
# ]

# # Ground truth sentences
# ground_truth_sentences = [
#     [
#         "Der Hund rennt.",
#         "Das Wetter ist gut.",
#         "Ein neues Auto.",
#         "Das Haus ist großartig.",
#     ]
# ]

# # Calculate the BLEU score using sacrebleu
# bleu_score = sacrebleu.corpus_bleu(translated_sentences, ground_truth_sentences).score

# print(f"BLEU score: {bleu_score}")
