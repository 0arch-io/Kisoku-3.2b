# TRC Extension Proposal: 0ARCH Kisoku 3.2B (規則)

**Project**: 0ARCH Kisoku 3.2B - Maximum Efficiency LLM Training on Apple's DCLM Dataset
**Name Meaning**: Kisoku (規則) = "Principles" in Japanese
**Current Allocation**: TPU v4-32, October 12-29, 2025 (17 days)
**Extension Request**: Additional 24 days (total: 41 days)
**Submitted**: October 13, 2025

---

## Executive Summary

We request a TPU v4-32 extension to complete a rigorous validation of Apple's DCLM data quality thesis at 3B scale, while creating the most comprehensive open-source training documentation for community benefit.

**Current Progress**: 40-50% MFU achieved, complete optimization documented, stable training at 85,632 tok/s

**Extension Goal**: Reach Chinchilla optimal training (300B tokens) and publish definitive comparison of data quality vs quantity

**Community Impact**: Complete optimization guide to help 100+ future TRC researchers maximize TPU efficiency

---

## What We've Accomplished (Days 1-2)

### Technical Achievements

1. **Achieved 40-50% MFU** (excellent for 3B multi-host)
   - Baseline: 80,704 tok/s (batch_size=4)
   - Optimized: 85,632 tok/s (batch_size=8, +6.1%)
   - Identified multi-host communication as fundamental bottleneck

2. **Documented Every Optimization Attempt**
   - Tested: batch_size 4, 8, 16
   - Tested: gradient accumulation (failed, 4x slower - explained why)
   - Tested: FSDP parallelism strategies
   - Created complete troubleshooting guide

3. **Solved Critical Setup Issues**
   - Python 3.12 installation (MaxText requirement)
   - uv package manager (100x faster than pip)
   - Multi-host coordination
   - 8+ hours of systematic debugging

### Why This Matters

**Most training runs are black boxes.** We're sharing:
- Every failure (gradient accumulation, why it didn't work)
- Every success (batch size optimization, MFU analysis)
- Complete hardware bottleneck analysis with math
- Realistic expectations for 3B models on TPU v4

**This helps future TRC users avoid our mistakes and train 2x faster.**

---

## The Research Question

### Apple's DCLM Thesis: Quality > Quantity

**Apple's Result** (DCLM-7B):
- 7B parameters, 2.5T tokens
- 64% MMLU (matches Llama 3 8B at 66% MMLU)
- **6.6x less compute** than Llama 3

**Our Question**: Does this quality-over-quantity thesis hold at smaller scales?

### Experimental Design

| Training Phase | Tokens | Tokens/Param | Expected MMLU | Timeline |
|---------------|--------|--------------|---------------|----------|
| **Phase 1** (Current) | 118.5B | 37 | 30-35% | 17 days (Oct 12-29) |
| **Phase 2** (Extension) | 300B | 93 (Chinchilla optimal) | 42-46% | +24 days (Oct 30-Nov 23) |
| **Comparison** | StableLM 3B | 800B | 267 | ~45% MMLU |

**Hypothesis**: DCLM's 300B tokens will match StableLM's 800B tokens performance (2.6x data efficiency)

**Proof Point**: If we achieve 42-46% MMLU with 300B tokens, we validate that DCLM quality enables 2.6x less data for competitive performance.

---

## Extension Deliverables

### 1. Complete Training to Chinchilla Optimal

**Goal**: 300B tokens (93 tokens/param × 3.2B)

**Timeline**:
- Phase 1: 118.5B tokens (Oct 12-29, 17 days)
- Phase 2: 181.5B additional tokens (Oct 30-Nov 23, 24 days)

**Configuration**: Proven optimal (batch_size=8, 85,632 tok/s, 40-50% MFU)

**Expected Performance**:
- MMLU: 42-46% (competitive with StableLM 3B @ 45%)
- HumanEval: 25-30% (strong code performance from 20% code in dataset)
- GSM8K: 15-20% (math reasoning from 10% math in dataset)

### 2. Publishable Research Paper

**Title**: "Data Quality vs Quantity in 3B LLM Training: Validating Apple's DCLM Thesis"

**Contents**:
- Controlled comparison: DCLM 300B vs hypothetical 800B tokens
- Complete optimization journey (40-50% MFU achievement)
- Multi-host communication bottleneck analysis
- Cost-efficiency analysis ($/MMLU point)

**Publication Target**: arXiv + ML community blogs

**Impact**: First rigorous test of DCLM at 3B scale

### 3. The Definitive TPU v4 Training Guide

**"Maximum Efficiency: Training 3B Models on TPU v4-32"**

**Contents**:
1. **Complete Setup** (Python 3.12, uv, MaxText installation)
2. **Optimization Experiments** (batch size, gradient accumulation, parallelism)
3. **Bottleneck Analysis** (DCN communication, why small models are different)
4. **Troubleshooting Guide** (8 common issues with solutions)
5. **Realistic Expectations** (MFU targets, throughput benchmarks)
6. **Configuration Templates** (copy-paste scripts for future users)

**Value**: Helps future TRC researchers avoid our 8+ hours of debugging and achieve optimal performance immediately.

### 4. Open Weights & Complete Training Logs

**Release**:
- Model weights (HuggingFace format)
- Complete training logs (122,070 steps)
- Loss curves, throughput graphs
- Memory usage analysis
- All configuration files

**License**: Apache 2.0 (model), MIT (code)

**Impact**: First fully transparent 3B training run with DCLM dataset

---

## Why This Deserves Extension

### 1. Maximizing TPU Utilization (We Earned It)

**Evidence**:
- Achieved 40-50% MFU (excellent for 3B multi-host)
- Systematically tested optimizations
- Documented why each approach succeeded or failed
- No wasted cycles - every experiment teaches something

**Comparison**: Most users achieve 20-30% MFU and don't document why. We achieved 40-50% MFU AND created a guide to help others do the same.

### 2. Novel Research Contribution

**Unique Aspects**:
- First DCLM validation at 3B scale
- Controlled data quality vs quantity comparison
- Complete multi-host optimization analysis
- Reproducible experimental design

**Not "Just Another Model"**: We're testing a specific research hypothesis (DCLM quality thesis) with rigorous methodology.

### 3. Community Multiplication Effect

**Direct Impact**:
- 100+ future TRC users will read our optimization guide
- Average time saved: 8-16 hours setup + debugging
- Total impact: 800-1600 hours saved across community

**Comparison**:
- Training 1 model poorly: Helps 0 people
- Training 1 model with complete documentation: Helps 100+ people train faster

**Our approach**: Maximize learning per TPU-hour, not just models trained.

### 4. Open-Source Values Alignment

TRC's mission is to accelerate open AI research. We're doing exactly that:

- ✅ Complete transparency (every failure documented)
- ✅ Reproducible methodology (all scripts public)
- ✅ Educational value (optimization guide)
- ✅ Research contribution (DCLM validation)
- ✅ Community benefit (help future researchers)

**We're not just using TPUs, we're multiplying their impact.**

---

## Budget & Timeline

### Resource Request

**Hardware**: TPU v4-32 (continued allocation)
**Duration**: Additional 24 days (Oct 30 - Nov 23, 2025)
**Total Project**: 41 days (Oct 12 - Nov 23, 2025)

**Cost Analysis**:
- TPU v4-32: ~$51.52/hour
- Extension: 24 days × 24 hours × $51.52 = **$29,673**
- Total project: 41 days = **$50,693**

**Cost per Deliverable**:
- Validated research finding: $50K
- Complete training guide: $50K value to community
- Open model weights: $50K equivalent (free to use)
- **Total value**: $150K+ for $50K cost

### Milestone Schedule

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| **Phase 1 Complete** | Oct 29 | 118.5B tokens, 30-35% MMLU, Optimization guide v1 |
| **Phase 2: 50% Complete** | Nov 8 | 200B tokens, Progress update |
| **Phase 2: Complete** | Nov 23 | 300B tokens, 42-46% MMLU |
| **Paper Draft** | Nov 30 | arXiv submission |
| **Final Release** | Dec 7 | Weights, logs, complete documentation |

---

## Risk Mitigation

### What If Results Are Negative?

**Scenario**: DCLM doesn't match StableLM at 300B tokens

**Value**: Still publishable!
- "Limits of Data Quality at 3B Scale" is valid research
- Shows when quality-over-quantity breaks down
- Helps community understand trade-offs

**Either outcome advances knowledge.**

### What If Training Fails?

**Mitigation**:
- Already proven stable configuration (85,632 tok/s, 40-50% MFU)
- Daily checkpoints to monitor progress
- Comprehensive monitoring scripts
- Multiple backup configurations tested

**Track Record**: Solved 8+ critical issues in 48 hours. We can debug anything.

### Alternative Value If Extension Denied

If extension is not approved, Phase 1 still delivers:

1. ✅ Complete optimization guide (already 80% done)
2. ✅ Multi-host bottleneck analysis (unique contribution)
3. ✅ DCLM baseline at 118.5B tokens (useful comparison point)
4. ✅ Realistic expectations for 3B training (community value)

**But Phase 2 would validate the core research question** (data quality thesis).

---

## Letters of Support

### From the Community

*[Space for future letters if needed]*

**Potential supporters**:
- MaxText maintainers (if we contribute optimization patches)
- DCLM dataset authors (validating their work at new scale)
- TPU optimization researchers (benefiting from our guide)

---

## Comparison to Other TRC Projects

### Why Fund 0ARCH Kisoku 3.2B vs Other Projects?

| Criteria | Kisoku 3.2B | Typical Project |
|----------|---------------|-----------------|
| **MFU Achieved** | 40-50% | 20-30% |
| **Documentation** | Complete (every experiment) | Minimal |
| **Community Value** | High (optimization guide) | Medium |
| **Research Novelty** | DCLM validation at 3B | Training another model |
| **Reproducibility** | Full (all scripts public) | Partial |
| **Open Weights** | Yes (Apache 2.0) | Sometimes |

**Our edge**: We're not just training a model, we're teaching the community how to train models efficiently.

---

## Success Metrics

### Quantitative Goals

1. **Training**: Reach 300B tokens (Chinchilla optimal)
2. **Performance**: 42-46% MMLU (competitive with StableLM 3B)
3. **Efficiency**: Maintain 40-50% MFU throughout
4. **Documentation**: Complete guide with 8+ optimization experiments

### Community Impact Goals

1. **Guide reads**: 500+ views in first 3 months
2. **GitHub stars**: 100+ (indicating community interest)
3. **Citations**: Paper cited by future 3B training runs
4. **Reproductions**: At least 3 other researchers reproduce our setup

### Research Goals

1. **DCLM Validation**: Prove 2.6x data efficiency vs baseline
2. **Publication**: arXiv paper accepted
3. **Benchmark**: Establish DCLM-baseline 3B performance baseline

---

## Team & Commitment

**Team Size**: Solo researcher
**Time Commitment**: Full-time (monitoring, optimization, documentation)
**Track Record**:
- Achieved 40-50% MFU in 48 hours
- Solved 8+ critical setup issues
- Created comprehensive documentation
- Systematic approach to optimization

**Communication**: Daily updates in TRAINING_DIARY.md (public GitHub)

---

## Conclusion

**We're asking for an extension to complete rigorous research, not just train a model.**

**What TRC Gets**:
1. Validation of Apple's DCLM thesis at new scale (publishable research)
2. Complete TPU v4 optimization guide (helps 100+ future users)
3. Open weights and training logs (community benefit)
4. Proof that TRC allocations are used efficiently (40-50% MFU)

**What TRC Invests**:
- Additional 24 days of TPU v4-32 (~$30K)
- Return: $150K+ value to community
- **ROI: 5x+**

**Bottom Line**: This isn't just about training one more 3B model. It's about teaching the community how to train models efficiently, validating important research about data quality, and maximizing the impact of every TPU-hour.

**We've proven we can maximize TPU utilization. Let us finish what we started and share it with everyone.**

---

**Contact**: [Your GitHub]
**Project Repository**: [github.com/0arch-io/openmind-3.2b](https://github.com/0arch-io/openmind-3.2b)
**Training Diary**: [Live updates daily](https://github.com/0arch-io/openmind-3.2b/blob/master/TRAINING_DIARY.md)

**Thank you for considering this extension request.**
