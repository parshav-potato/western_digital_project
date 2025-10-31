# Performance Tips & Troubleshooting

## 🚀 Preventing Crashes on Large PDFs

### Problem
Processing large PDFs (like the 700-page Bishop book) can:
- ❌ Consume too much memory
- ❌ Take hours to complete
- ❌ Use significant API credits (each element = 1 LLM call)
- ❌ Crash your computer

### Solution: Optimized Processing Options

#### 1. **Limit Number of Elements** (Recommended for testing)

```python
# In the notebook, set:
MAX_ELEMENTS = 50  # Process only first 50 text chunks
SKIP_IMAGES = True  # Skip image processing

documents, raw_texts = pdf_processor.process_pdf(
    pdf_path=PDF_FILE,
    max_elements=MAX_ELEMENTS,
    skip_images=SKIP_IMAGES
)
```

**Time**: ~2-5 minutes  
**Cost**: ~$0.10-0.50 (50 elements × ~1-2¢ each)

#### 2. **Quick Test Settings**

For initial testing with any large PDF:

```python
MAX_ELEMENTS = 20   # Very quick test
SKIP_IMAGES = True  # Always skip for testing
```

**Time**: ~1-2 minutes  
**Cost**: ~$0.05-0.20

#### 3. **Full Processing**

Only use after confirming the workflow works:

```python
MAX_ELEMENTS = None  # Process everything
SKIP_IMAGES = False  # Include images
```

**Time**: 30+ minutes for 700-page book  
**Cost**: $5-15+ depending on content

### New Parameters

#### `max_elements` (int or None)
- Limits how many text/table elements to process
- `None` = process all elements
- `50` = process first 50 elements
- Useful for testing workflow before full run

#### `skip_images` (bool)
- `True` = Skip image extraction and analysis
- `False` = Process images (requires vision-capable model)
- Images are slowest to process (vision API calls)

## 📊 Cost Estimation

### Per Element Costs (Approximate)

| Element Type | API Calls | Cost per Element |
|-------------|-----------|------------------|
| Text chunk | 1 | $0.01-0.02 |
| Table | 1 | $0.01-0.02 |
| Image | 2 (vision + context) | $0.05-0.10 |

### Example Calculations

**Bishop Book (~700 pages)**:
- Estimated elements: ~1,500-2,000 text chunks
- Estimated tables: ~50-100
- Estimated images: ~200-300

**Full Processing**:
- Text: 1,500 × $0.015 = $22.50
- Tables: 75 × $0.015 = $1.13
- Images: 250 × $0.075 = $18.75
- **Total: ~$40-50**

**Limited Processing (50 elements, no images)**:
- 50 × $0.015 = **$0.75**

## 💡 Recommended Workflow

### Step 1: Test with Small Sample
```python
MAX_ELEMENTS = 20
SKIP_IMAGES = True
```

Verify:
- ✅ PDF extraction works
- ✅ Summaries are generated
- ✅ No errors occur

### Step 2: Medium Test
```python
MAX_ELEMENTS = 100
SKIP_IMAGES = True
```

Test:
- ✅ RAPTOR clustering works
- ✅ Vector store creation succeeds
- ✅ Queries return good results

### Step 3: Full Run (Optional)
```python
MAX_ELEMENTS = None
SKIP_IMAGES = False  # Only if you need images
```

## 🛠️ Error Handling

The updated code includes:

### Try-Catch Blocks
```python
try:
    # Process element
except Exception as e:
    print(f"⚠️ Error processing element {i}: {e}")
    continue  # Skip and move to next
```

### Progress Indicators
```
Processing 10/50 elements...
Processing 20/50 elements...
```

### Automatic Skipping
- Invalid images automatically skipped
- Malformed elements logged and skipped
- Processing continues even if individual elements fail

## 🔍 Monitoring Progress

Watch for these outputs:

```
📄 Extracting elements from PDF: ...
✅ Extracted 1,234 elements

📝 Processing text and tables...
⚠️  Processing only first 50 elements (of 1,234 total)
  Processed 10/50 elements...
  Processed 20/50 elements...
✅ Processed 50 text/table elements

⏭️  Skipping image processing (skip_images=True)

🎉 Total: 50 documents extracted
```

## 🎯 For Different PDF Sizes

### Small PDFs (<50 pages)
```python
MAX_ELEMENTS = None  # Process all
SKIP_IMAGES = False  # Include images
```

### Medium PDFs (50-200 pages)
```python
MAX_ELEMENTS = 200   # Sample
SKIP_IMAGES = True   # Skip images first
```

### Large PDFs (200+ pages)
```python
MAX_ELEMENTS = 50    # Small sample
SKIP_IMAGES = True   # Definitely skip
```

## 💰 Managing API Costs

### OpenAI Costs
- GPT-4o: ~$2.50 / 1M input tokens, ~$10 / 1M output tokens
- GPT-4-turbo: ~$10 / 1M input tokens, ~$30 / 1M output tokens
- Vision: +$0.01-0.02 per image

### Gemini Costs
- Gemini Pro: Free tier available!
- 15 RPM free
- Good for testing

### Cost-Saving Tips

1. **Use Gemini for testing**:
   ```python
   config = Config(llm_provider="gemini")
   ```

2. **Process incrementally**:
   - Start with 20 elements
   - Increase gradually
   - Save results frequently

3. **Skip images when possible**:
   - Images are 5-10x more expensive
   - Only process if needed

4. **Cache results**:
   ```python
   # After processing, save documents
   import pickle
   with open('processed_docs.pkl', 'wb') as f:
       pickle.dump((documents, raw_texts), f)
   
   # Load later without reprocessing
   with open('processed_docs.pkl', 'rb') as f:
       documents, raw_texts = pickle.load(f)
   ```

## 🔧 Troubleshooting

### "Out of Memory" Error
- Reduce `MAX_ELEMENTS` to 20
- Set `SKIP_IMAGES = True`
- Close other applications
- Restart notebook kernel

### "API Rate Limit" Error
- Reduce processing speed
- Use smaller `MAX_ELEMENTS`
- Switch to Gemini (higher free tier)

### "Computer Freezing"
- Lower `MAX_ELEMENTS` to 10-20
- Enable `SKIP_IMAGES`
- Monitor RAM usage
- Process during off-peak hours

### "Taking Too Long"
- Check progress indicators
- Reduce `MAX_ELEMENTS`
- Skip images initially
- Consider processing overnight

## ✅ Best Practices

1. **Always test small first**: Start with 20 elements
2. **Monitor costs**: Check API usage dashboard
3. **Save intermediate results**: Don't lose progress
4. **Use Gemini for testing**: Free tier is generous
5. **Process incrementally**: Don't try to do everything at once

---

**Remember**: You can always process more elements later. Start small, verify it works, then scale up!
