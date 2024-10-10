import polars as pl

df=pl.read_parquet("curated/weighted_compiled_topk.parquet")



n_sequences=150_000

def mutate_rna_nucleotide(sequence, position):
    # Check if the position is within the bounds of the sequence
    if position < 0 or position >= len(sequence):
        return "Invalid position"
    
    # Define a dictionary to store the complements of nucleotides
    complement_dict = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    
    # Get the nucleotide at the specified position
    current_nucleotide = sequence[position]
    
    # Check if the current nucleotide is in the dictionary
    if current_nucleotide in complement_dict:
        # Replace the nucleotide at the specified position with its complement
        mutated_sequence = sequence[:position] + complement_dict[current_nucleotide] + sequence[position+1:]
        return mutated_sequence
    else:
        return "Invalid nucleotide"

sequences=[]
ids=[]
for i in range(len(df)):
    sequence=df['sequence'][i]
    chromosome=df['chromosome'][i]
    position=df['start_position'][i]
    pk_string=df['pseudo_knot_string'][i]


    # sequences.append(sequence)
    # ids.append(chromosome+f"_{position}_p")
    for j,s in enumerate(pk_string):
        if s=='X':
            if len(sequences)==n_sequences:
                break
            new_sequence=mutate_rna_nucleotide(sequence, j)
            id=chromosome+f"_{position}_{j}"
            sequences.append(new_sequence)
            ids.append(id)

    if len(sequences)==n_sequences:
        break
    print(i)


submission=pl.DataFrame()
submission=submission.with_columns(pl.Series('sequence',sequences))
submission=submission.with_columns(pl.Series('sequence_id',ids))

submission.write_csv('submission.csv')