# BERT Poetic

```
i was the shadow of the waxwing slain
i was the shadow of the waxwing slow
i was the shadadow of the waxwing slow
i saw the shadadow of the waxwing slow
i saw the shoadadow of the waxwing slow
i saw the shobadow of the waxwing slow
i saw the shobadow of the waxrowing slow
i saw the brobadow of the waxrowing slow
i saw the brobadowing of the waxrowing slow
i saw the brobadowing of the walrowing slow
i saw the brobadowing of the solrowing slow
i saw the brobadowing of the old solrowing slow
i saw the broblowing of the old solrowing slow
i saw the broblnowing of the old solrowing slow
i saw the broblnowing of the older solrowing slow
i saw the spoblnowing of the older solrowing slow
i saw the spoblnowing of the water solrowing slow
i saw in the spoblnowing of the water solrowing slow
i saw in the spoblnow spring of the water solrowing slow
i saw in the spoblnow spring of the water sorrowing slow
i saw in the spoblnow spring of all the water sorrowing slow
i saw in the spoblnow spring of all, the water sorrowing slow
i saw in the spokenow spring of all, the water sorrowing slow
i saw in the sponkenow spring of all, the water sorrowing slow
i am in the sponkenow spring of all, the water sorrowing slow
i i am in the sponkenow spring of all, the water sorrowing slow
i i am in the sponkenow spring of all, the water sorrow to slow
i i am in the sponkenow spring of all things, the water sorrow to slow
i i am in the sponkenow springing all things, the water sorrow to slow
```

## What's this?
A collection of tools to explore poetry generation
with a neural network transformer language model.

## How does it work?

1. Get a whole bunch of poetry from the [Gutenberg,
   dammit](https://github.com/aparrish/gutenberg-dammit) dataset.

2. Use the poetry to train a [word-piece
   tokenizer](https://github.com/huggingface/tokenizers). A word-piece
   tokenizer splits words into a sequence of words or pieces of words. The smaller
   the the vocabulary size, the smaller the pieces of words have to be. I want to
   train a model to pay attention to meter and rhyme rather than syntax and
   semantics, so I use a very small vocabulary size of 1000. (This is probably a
   good parameter to experiment on!)

3. Pre-train a [BERT](https://www.aclweb.org/anthology/N19-1423.pdf) masked
   language model task on lines of poetry. The model replaces a random token
   in each example with the `[MASK]` token, then learns to predict what word
   was replaced based on context.

4. The real value of this pre-trained language model will be in refining it
   for further tasks related to poetry generation, such as metrical classification
   and rhyme prediction. This is COMING SOON.

5. But in the mean time, it's fun make it generate text by randomly replacing
   tokens in an input text with a `[MASK]` token or randomly inserting a `[MASK]`
   token, getting the model's prediction for the likliest token to fill that
   position, inserting that predicted token, and finally repeating the process
   with the new text.
 
