# Open Ingest Pipeline - Mermaid Flowchart

This document is a visual reference for the end-to-end ingestion DAG. The pipeline
flows: **upload → file conversion → routing → OCR / layout → optional VLM enrichment
→ optional structured extraction → assembled output**. Each stage is a Tensorlake
`@function` that runs in its own Firecracker microVM, with S3 as the durable substrate
between steps — so any stage can be retried, resumed, or inspected in isolation.

The diagrams below show the full graph and per-branch detail; use them to find the
function that owns a given behavior before diving into `src/tensorlake_docai/`.

A few terms used in the diagrams:

- **`skip_ocr`** — a per-request flag on `StructuredExtractionRequest`. When
  `True`, the pipeline skips OCR entirely and feeds page images directly to a
  vision LLM (Gemini) for structured extraction. Useful for visually dense
  documents where OCR misses layout cues.
- **Chunking strategies** — `none` (whole doc), `page` (one chunk per page),
  `section` / `fragment` (layout-driven), or `patterns` (regex boundaries). Set
  per-extraction via `StructuredExtractionRequest.chunking_strategy`.

## How to View/Edit This Diagram

1. **GitHub/GitLab**: Paste this code in a .md file - it will render automatically
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online Editor**: https://mermaid.live/ - paste and edit in real-time
4. **Notion**: Use `/code` block and select "Mermaid"
5. **Draw.io**: Import Mermaid code directly
6. **Obsidian**: Native Mermaid support

---

## Complete Pipeline Flowchart

```mermaid
flowchart TD
    Start([User Upload]) --> FileConv[FILE_CONVERTOR<br/>normalize_file_type_and_upload]
    
    FileConv --> Detect{Detect<br/>File Type}
    
    Detect --> P7M{P7M<br/>File?}
    P7M -->|Yes| ExtractP7M[Extract P7M Content<br/>OpenSSL]
    ExtractP7M --> Detect
    P7M -->|No| FileType{File Type?}
    
    FileType -->|DOC| ConvertDOC[Convert DOC to DOCX<br/>LibreOffice]
    ConvertDOC --> ProcessDOCX
    
    FileType -->|DOCX| ProcessDOCX[Process DOCX<br/>Extract structure + bboxes<br/>Generate PDF]
    ProcessDOCX --> ValidateQuota
    
    FileType -->|Excel/CSV| ProcessExcel[Process Excel<br/>Convert to HTML tables<br/>Create PageLayout]
    ProcessExcel --> ValidateQuota
    
    FileType -->|Text| ProcessText[Process Text Files<br/>Create PageLayout<br/>No OCR needed]
    ProcessText --> ValidateQuota
    
    FileType -->|PPT/PPTX<br/>RTF<br/>Other| ConvertPDF[Convert to PDF<br/>LibreOffice soffice]
    ConvertPDF --> ValidateQuota
    
    FileType -->|PDF/Image| ValidateQuota[Validate Quotas<br/>Count Pages]
    
    ValidateQuota --> Route{Routing<br/>Decision}
    
    %% Text File Routes
    Route -->|Text File| TextRoute{Has<br/>Processing?}
    TextRoute -->|Page Classification| VLMText[VLMExtractionTask<br/>Text-only classification]
    TextRoute -->|Structured Extraction| SEText[StructuredExtraction<br/>LLM extraction]
    TextRoute -->|None| OutText[OutputFormatter]
    
    %% Skip OCR Route
    Route -->|Skip OCR=True| VLMDirect[VLMExtractionTask<br/>Direct VLM Processing]
    
    %% OCR Routes
    Route -->|Need OCR| OCRSelect{OCR Model<br/>Selection}
    
    OCRSelect -->|azure-di<br/>Azure cloud| Azure[FullPageAzureTask<br/>Azure Document Intelligence<br/>- Layout + Markdown<br/>- Tables/Forms<br/>- Figure extraction<br/>- Native PDF support]
    
    OCRSelect -->|textract<br/>AWS cloud| Textract[FullPageTextractTask<br/>AWS Textract<br/>- Layout + Markdown<br/>- Tables/Forms<br/>- Figure extraction<br/>- Native PDF support]
    
    OCRSelect -->|gemini<br/>Google cloud| Gemini[FullPageGeminiTask<br/>Google Gemini VLM<br/>- Semantic tags<br/>- Tables/Figures<br/>- Native PDF support]
    
    OCRSelect -->|dots-ocr<br/>CUDA GPU| DotsOCR[DotsOCRTask<br/>DotsOCR on CUDA GPU worker<br/>- Layout + Markdown<br/>- Two-stage Ovis figure OCR<br/>- Barcode detection]
    
    %% Optional Header Correction
    Azure --> HeaderOpt{Header<br/>Correction?}
    Textract --> HeaderOpt
    Gemini --> HeaderOpt
    DotsOCR --> HeaderOpt
    
    HeaderOpt -->|Yes| HeaderCorr[Header Correction<br/>OpenAI GPT]
    HeaderOpt -->|No| PostOCR
    HeaderCorr --> PostOCR{Post-OCR<br/>Routing}
    
    %% Post-OCR Routing
    PostOCR -->|No Further<br/>Processing| OutOCR[OutputFormatter]
    PostOCR -->|VLM Tasks<br/>Needed| VLMTask[VLMExtractionTask]
    PostOCR -->|Structured<br/>Extraction Only| SETask[StructuredExtraction]
    
    %% VLM Task Details
    VLMTask --> VLMProcess[VLM Batch Processing:<br/>1. Table Summarization<br/>2. Figure Summarization<br/>3. Signature Detection<br/>4. Page Classification<br/>5. Structured Extraction when skip_ocr=True]
    VLMDirect --> VLMProcess
    VLMText --> OutVLM
    
    VLMProcess --> VLMRoute{More<br/>Processing?}
    VLMRoute -->|Structured<br/>Extraction| SEFromVLM[StructuredExtraction<br/>LLM-based]
    VLMRoute -->|Done| OutVLM[OutputFormatter]
    
    %% Structured Extraction Details
    SETask --> SEProcess[Structured Extraction:<br/>- Model: OpenAI/Claude/Gemini<br/>- Chunking strategies<br/>- Citation tracking<br/>- Dense table splitting<br/>- Parallel processing]
    SEText --> SEProcess
    SEFromVLM --> SEProcess
    
    SEProcess --> OutSE[OutputFormatter]
    
    %% Final Output
    OutText --> Final[Final Output:<br/>- Aggregate tokens<br/>- Format response<br/>- ParsedDocumentRef]
    OutOCR --> Final
    OutVLM --> Final
    OutSE --> Final
    
    Final --> End([Return to User])
    
    %% Styling
    classDef entryPoint fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    classDef ocrModel fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef vlmTask fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef structTask fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:3px,color:#000
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000
    
    class FileConv entryPoint
    class Azure,Textract,Gemini,DotsOCR ocrModel
    class VLMTask,VLMDirect,VLMProcess,VLMText vlmTask
    class SETask,SEText,SEFromVLM,SEProcess structTask
    class OutText,OutOCR,OutVLM,OutSE,Final output
    class Detect,P7M,FileType,Route,TextRoute,OCRSelect,HeaderOpt,PostOCR,VLMRoute decision
```

---

## Simplified High-Level Flow

```mermaid
flowchart LR
    A[File Upload] --> B[File Convertor]
    B --> C{File Type?}
    
    C -->|Text/Excel/DOCX| D[Direct Processing<br/>No OCR]
    C -->|PDF/Image| E{OCR Model}
    C -->|PPT/RTF/Other| CONV[Convert to PDF]
    
    CONV --> E
    
    E -->|azure-di| F1[Azure Document Intelligence]
    E -->|textract| F2[AWS Textract]
    E -->|gemini| F3[Google Gemini VLM]
    E -->|dots-ocr| F4[DotsOCR on CUDA GPU worker]
    
    D --> G{Processing<br/>Needed?}
    F1 --> G
    F2 --> G
    F3 --> G
    F4 --> G
    
    G -->|VLM Tasks| H[VLM Processing<br/>• Table/Figure: OpenAI<br/>• Signatures: Textract<br/>• Page Class: OpenAI<br/>• Structured Extraction skip_ocr: Gemini]
    G -->|LLM SE Only| J
    G -->|None| K
    
    H --> I{LLM SE<br/>Needed?}
    
    I -->|Yes| J[LLM Extraction<br/>OpenAI/Claude/Gemini<br/>• Chunking strategies<br/>• Citation tracking]
    I -->|No| K[Output Formatter]
    
    J --> K
    K --> L[API Response]
    
    classDef entry fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    classDef ocr fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef vlm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef llm fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef out fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#000
    classDef convert fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000
    
    class A,B entry
    class F1,F2,F3,F4 ocr
    class H vlm
    class J llm
    class K,L out
    class CONV convert
```

---

## VLM Extraction Task Detail

```mermaid
flowchart TD
    VLMStart[VLMExtractionTask Start] --> BatchCreate[Create Image Batches<br/>Memory-efficient processing]
    
    BatchCreate --> Batch1{Batch Processing}
    
    Batch1 -->|For each batch| TableSum[Table Summarization<br/>OpenAI VLM<br/>Crop + Describe]
    Batch1 -->|For each batch| FigSum[Figure Summarization<br/>OpenAI VLM<br/>Crop + Describe]
    Batch1 -->|For each batch| SigDet[Signature Detection<br/>AWS Textract<br/>Parallel ThreadPool]
    Batch1 -->|For each batch| PageClass[Page Classification<br/>OpenAI VLM<br/>Multi-label support]
    Batch1 -->|For each batch| SkipOCRSE[Structured Extraction<br/>skip_ocr=True<br/>Gemini VLM]
    
    TableSum --> UpdateElements[Update PageLayout<br/>Elements In-Place]
    FigSum --> UpdateElements
    SigDet --> UpdateElements
    PageClass --> UpdateElements
    SkipOCRSE --> StoreResults[Store in<br/>structured_outputs_by_page]
    
    UpdateElements --> MoreBatches{More<br/>Batches?}
    StoreResults --> MoreBatches
    
    MoreBatches -->|Yes| Batch1
    MoreBatches -->|No| Deferred{Deferred SE<br/>None-chunked?}
    
    Deferred -->|Yes| DeferredSE[Process Deferred<br/>Structured Extraction<br/>Document-level across pages]
    Deferred -->|No| TokenAgg[Aggregate Token Usage<br/>- Summarization tokens<br/>- Extraction tokens]
    
    DeferredSE --> TokenAgg
    TokenAgg --> VLMEnd[Route to Next Stage]
    
    classDef vlm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef process fill:#e3f2fd,stroke:#0277bd,stroke-width:2px,color:#000
    
    class VLMStart,VLMEnd vlm
    class TableSum,FigSum,SigDet,PageClass,SkipOCRSE,UpdateElements,StoreResults,DeferredSE,TokenAgg process
```

---

## Structured Extraction Task Detail

```mermaid
flowchart TD
    SEStart[StructuredExtraction Start] --> CheckVLM{VLM outputs<br/>exist?}
    
    CheckVLM -->|Yes| SkipLLM[Skip LLM Extraction<br/>Use VLM results]
    CheckVLM -->|No| GetRequests[Get structured_extraction_requests]
    
    GetRequests --> FilterPages{Filter by<br/>page_classes?}
    
    FilterPages -->|Yes| FilterLogic[Filter pages matching<br/>specified classes]
    FilterPages -->|No| AllPages[Use all pages]
    
    FilterLogic --> ChunkStrategy{Chunking<br/>Strategy?}
    AllPages --> ChunkStrategy
    
    ChunkStrategy -->|None| WholeDoc[Process whole document]
    ChunkStrategy -->|Page| PerPage[Process per page]
    ChunkStrategy -->|Section| PerSection[Process per section]
    ChunkStrategy -->|Fragment| PerFragment[Process per fragment]
    ChunkStrategy -->|Pattern| ByPattern[Process by regex patterns]
    
    WholeDoc --> PrepText[Prepare Text Content]
    PerPage --> PrepText
    PerSection --> PrepText
    PerFragment --> PrepText
    ByPattern --> PrepText
    
    PrepText --> Citations{Citations<br/>enabled?}
    
    Citations -->|Yes| AddRefs[Add Citation References<br/>ref_id tracking]
    Citations -->|No| PageMarkers{Add page<br/>markers?}
    
    AddRefs --> CheckDense
    PageMarkers -->|Yes| AddMarkers[Add page boundary markers]
    PageMarkers -->|No| CheckDense
    AddMarkers --> CheckDense{Dense table<br/>content?}
    
    CheckDense -->|Yes CSV/Excel| SplitTable[Split Dense Table<br/>Row-based chunking<br/>Parallel processing]
    CheckDense -->|No| ModelCall[LLM API Call<br/>OpenAI/Claude/Gemini]
    
    SplitTable --> MergeResults[Merge Chunked Results]
    ModelCall --> ParseJSON[Parse JSON Response]
    MergeResults --> ParseJSON
    
    ParseJSON --> ResolveCite{Resolve<br/>citations?}
    
    ResolveCite -->|Yes| ValidateCite[Validate page references<br/>Clean citation data]
    ResolveCite -->|No| StoreResult[Store Structured Data<br/>by page/chunk key]
    
    ValidateCite --> StoreResult
    StoreResult --> MoreReq{More<br/>requests?}
    
    MoreReq -->|Yes| GetRequests
    MoreReq -->|No| AggTokens[Aggregate Token Usage<br/>Input + Output tokens]
    
    AggTokens --> SEEnd[Route to OutputFormatter]
    SkipLLM --> SEEnd
    
    classDef struct fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef process fill:#e3f2fd,stroke:#0277bd,stroke-width:2px,color:#000
    
    class SEStart,SEEnd struct
    class PrepText,AddRefs,AddMarkers,SplitTable,ModelCall,MergeResults,ParseJSON,ValidateCite,StoreResult,AggTokens process
```

---

## OCR Model Comparison Table

| `ocr_model` | Provider | Speed | Layout | Tables | Figures | Forms | Special Features |
|-------------|----------|-------|--------|--------|---------|-------|------------------|
| `dots-ocr` | DotsOCR on CUDA GPU worker | Fast | ✓ | ✓ | ✓ | - | Custom prompts, Barcodes, two-stage Ovis figure OCR |
| `azure-di`  | Azure Document Intelligence | Fast | ✓ | ✓ | ✓ | ✓ | Native PDF, Cell bboxes, 100pg chunks |
| `textract` | AWS Textract | Fast | ✓ | ✓ | ✓ | ✓ | Native PDF, S3 async, Signatures |
| `gemini` | Google Gemini VLM | Medium | ✓ | ✓ | ✓ | - | Native PDF, Semantic tagging |

---

## File Type Processing Matrix

```mermaid
flowchart TD
    Input[File Input] --> Type{File Type?}
    
    Type -->|PDF| PDF[PDF Processing<br/>✓ All OCR models<br/>✓ Page selection]
    
    Type -->|Images<br/>jpg/png/tiff| IMG[Image Processing<br/>✓ All OCR models<br/>✓ RGBA conversion<br/>✓ Multi-page TIFF]
    
    Type -->|DOCX| DOCX[DOCX Processing<br/>✓ Structure extraction<br/>✓ Tracked changes<br/>✓ PDF conversion<br/>✗ No OCR needed]
    
    Type -->|DOC| DOC[DOC Processing<br/>1. Convert to DOCX<br/>2. Process as DOCX<br/>✗ No OCR needed]
    
    Type -->|Excel<br/>xlsx/xls/xlsm| EXCEL[Excel Processing<br/>✓ Multi-sheet support<br/>✓ HTML tables<br/>✓ Markdown conversion<br/>✗ No OCR needed]
    
    Type -->|CSV| CSV[CSV Processing<br/>✓ Text table format<br/>✓ Dense table splitting<br/>✗ No OCR needed]
    
    Type -->|Text<br/>txt/html/xml/md| TXT[Text Processing<br/>✓ UTF-8 encoding<br/>✓ Direct PageLayout<br/>✗ No OCR needed]
    
    Type -->|PPT/PPTX<br/>RTF<br/>Other Office| CONVERT[Convert to PDF<br/>1. LibreOffice soffice<br/>2. Process as PDF<br/>✓ Then needs OCR]
    
    Type -->|P7M| P7M[P7M Processing<br/>1. Extract with OpenSSL<br/>2. Detect inner type<br/>3. Process accordingly]
    
    P7M --> Type
    CONVERT --> PDF
    
    classDef needsOCR fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef noOCR fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef converter fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000
    
    class PDF,IMG needsOCR
    class DOCX,DOC,EXCEL,CSV,TXT noOCR
    class CONVERT converter
```
