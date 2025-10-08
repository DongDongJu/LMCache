Using Different Caching Policies
===================================

LMCache supports multiple caching policies.

For example, to use LRU, you can set the environment variable ``LMCACHE_CACHE_POLICY=LRU`` or set it in the configuration file with ``cache_policy="LRU"``.

To enable the SIEVE family policies, run::

   export LMCACHE_CACHE_POLICY=SIEVE

   # or segmented SIEVE with a probation window
   export LMCACHE_CACHE_POLICY=SIEVE_SLRU

   # or Prefill/Decode-Guarded SIEVE with a decode hotset guard
   export LMCACHE_CACHE_POLICY=SIEVE_PDG

   # or CLOCK with an Evictable Candidate List and decode guard TTL
   export LMCACHE_CACHE_POLICY=CLOCK_ECL

Currently, LMCache supports "LRU" (Least Recently Used), "MRU" (Most Recently Used), "LFU" (Least Frequently Used), "FIFO" (First-In-First-Out), and "SIEVE" (single-queue scan with multi-bit reuse tracking that balances recency and frequency). `SIEVE_SLRU` adds a small MRU probation window in front of SIEVE to absorb large sequential prefills before promoting frequently reused decode entries. `SIEVE_PDG` (Prefill/Decode-Guarded SIEVE) extends this idea with a dedicated decode guard buffer that keeps active decode KV blocks out of the scanning queue, while probation continues to absorb wide prefills. `CLOCK_ECL` is a CLOCK-style second-chance policy with an Evictable Candidate List for constant-time victim selection and a short decode guard TTL that protects active conversations. All SIEVE variants and `CLOCK_ECL` follow the same Prefill/Decode (PD) constraints as the other options and require no additional configuration.
