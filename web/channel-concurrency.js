"use strict";

const DEFAULT_CHANNEL = "api";

function asPositiveInteger(value, fallback, minimum = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.floor(parsed)) : fallback;
}

class ChannelConcurrencyManager {
  constructor(options = {}) {
    this.maxConcurrency = asPositiveInteger(options.maxConcurrency, 4, 1);
    this.queueTimeoutMs = asPositiveInteger(options.queueTimeoutMs, 120_000, 1);
    this.sessionQueueLimit = asPositiveInteger(options.sessionQueueLimit, 2, 0);
    this.fallbackChannel = String(options.fallbackChannel || DEFAULT_CHANNEL).toLowerCase();
    this.channels = new Map();
    this.schedule = [];

    const configuredChannels = options.channels || {};
    for (const [name, config] of Object.entries(configuredChannels)) {
      const channel = String(name || "").trim().toLowerCase();
      if (!channel) continue;
      const queueLimit = asPositiveInteger(config?.queueLimit, 8, 0);
      const weight = asPositiveInteger(config?.weight, 1, 1);
      this.channels.set(channel, {
        name: channel,
        queueLimit,
        queue: [],
        active: 0,
        stats: {
          accepted: 0,
          started: 0,
          completed: 0,
          rejectedFull: 0,
          rejectedSession: 0,
          rejectedDuplicate: 0,
          timedOut: 0,
          totalWaitMs: 0,
          maxWaitMs: 0,
        },
      });
      for (let index = 0; index < weight; index += 1) this.schedule.push(channel);
    }

    if (!this.channels.has(this.fallbackChannel)) {
      this.channels.set(this.fallbackChannel, {
        name: this.fallbackChannel,
        queueLimit: 8,
        queue: [],
        active: 0,
        stats: {
          accepted: 0,
          started: 0,
          completed: 0,
          rejectedFull: 0,
          rejectedSession: 0,
          rejectedDuplicate: 0,
          timedOut: 0,
          totalWaitMs: 0,
          maxWaitMs: 0,
        },
      });
      this.schedule.push(this.fallbackChannel);
    }

    this.active = 0;
    this.activeJobs = new Map();
    this.activeSessions = new Set();
    this.knownRequestIds = new Set();
    this.cursor = 0;
    this.nextJobId = 1;
  }

  normalizeChannel(value) {
    const channel = String(value || "").trim().toLowerCase();
    return this.channels.has(channel) ? channel : this.fallbackChannel;
  }

  normalizeSession(value) {
    return String(value || "").trim().slice(0, 240);
  }

  normalizeRequestId(value) {
    return String(value || "").trim().slice(0, 240);
  }

  queuedCount() {
    let total = 0;
    for (const state of this.channels.values()) total += state.queue.length;
    return total;
  }

  queuedForSession(sessionId) {
    if (!sessionId) return 0;
    let total = 0;
    for (const state of this.channels.values()) {
      total += state.queue.reduce((count, job) => count + Number(job.sessionId === sessionId), 0);
    }
    return total;
  }

  acquire(meta = {}) {
    const channel = this.normalizeChannel(meta.channel);
    const state = this.channels.get(channel);
    const requestId = this.normalizeRequestId(meta.requestId);
    const sessionId = this.normalizeSession(meta.sessionId);
    const canStartImmediately = (
      this.active < this.maxConcurrency
      && this.queuedCount() === 0
      && (!sessionId || !this.activeSessions.has(sessionId))
    );

    if (requestId && this.knownRequestIds.has(requestId)) {
      state.stats.rejectedDuplicate += 1;
      return Promise.resolve(null);
    }
    if (!canStartImmediately && state.queue.length >= state.queueLimit) {
      state.stats.rejectedFull += 1;
      return Promise.resolve(null);
    }
    if (
      !canStartImmediately
      && sessionId
      && this.queuedForSession(sessionId) >= this.sessionQueueLimit
    ) {
      state.stats.rejectedSession += 1;
      return Promise.resolve(null);
    }

    return new Promise((resolve) => {
      const job = {
        id: this.nextJobId++,
        channel,
        requestId,
        sessionId,
        createdAt: Date.now(),
        wasQueued: !canStartImmediately,
        resolve,
        timer: null,
      };
      state.stats.accepted += 1;
      state.queue.push(job);
      if (requestId) this.knownRequestIds.add(requestId);
      if (job.wasQueued) {
        job.timer = setTimeout(() => this.expire(job), this.queueTimeoutMs);
        job.timer.unref?.();
      }
      this.pump();
    });
  }

  expire(job) {
    const state = this.channels.get(job.channel);
    const index = state.queue.findIndex((candidate) => candidate.id === job.id);
    if (index < 0) return;
    state.queue.splice(index, 1);
    state.stats.timedOut += 1;
    if (job.requestId) this.knownRequestIds.delete(job.requestId);
    job.resolve(null);
    this.pump();
  }

  takeNextEligible() {
    for (let attempt = 0; attempt < this.schedule.length; attempt += 1) {
      const channel = this.schedule[this.cursor];
      this.cursor = (this.cursor + 1) % this.schedule.length;
      const state = this.channels.get(channel);
      const index = state.queue.findIndex((job) => (
        !job.sessionId || !this.activeSessions.has(job.sessionId)
      ));
      if (index >= 0) return state.queue.splice(index, 1)[0];
    }
    return null;
  }

  pump() {
    while (this.active < this.maxConcurrency) {
      const job = this.takeNextEligible();
      if (!job) return;
      this.start(job);
    }
  }

  start(job) {
    const state = this.channels.get(job.channel);
    if (job.timer) clearTimeout(job.timer);
    const waitMs = Math.max(0, Date.now() - job.createdAt);
    this.active += 1;
    state.active += 1;
    state.stats.started += 1;
    state.stats.totalWaitMs += waitMs;
    state.stats.maxWaitMs = Math.max(state.stats.maxWaitMs, waitMs);
    if (job.sessionId) this.activeSessions.add(job.sessionId);
    this.activeJobs.set(job.id, job);

    let released = false;
    job.resolve(Object.freeze({
      channel: job.channel,
      requestId: job.requestId,
      sessionId: job.sessionId,
      queued: job.wasQueued,
      waitMs,
      release: () => {
        if (released) return;
        released = true;
        this.release(job.id);
      },
    }));
  }

  release(jobId) {
    const job = this.activeJobs.get(jobId);
    if (!job) return;
    const state = this.channels.get(job.channel);
    this.activeJobs.delete(jobId);
    this.active = Math.max(0, this.active - 1);
    state.active = Math.max(0, state.active - 1);
    state.stats.completed += 1;
    if (job.sessionId) this.activeSessions.delete(job.sessionId);
    if (job.requestId) this.knownRequestIds.delete(job.requestId);
    this.pump();
  }

  snapshot() {
    const channels = {};
    let queueLimit = 0;
    for (const [name, state] of this.channels.entries()) {
      queueLimit += state.queueLimit;
      channels[name] = {
        active: state.active,
        queued: state.queue.length,
        queue_limit: state.queueLimit,
        accepted: state.stats.accepted,
        started: state.stats.started,
        completed: state.stats.completed,
        rejected_full: state.stats.rejectedFull,
        rejected_session: state.stats.rejectedSession,
        rejected_duplicate: state.stats.rejectedDuplicate,
        timed_out: state.stats.timedOut,
        average_wait_ms: state.stats.started
          ? Math.round(state.stats.totalWaitMs / state.stats.started)
          : 0,
        max_wait_ms: state.stats.maxWaitMs,
      };
    }
    return {
      active: this.active,
      limit: this.maxConcurrency,
      queued: this.queuedCount(),
      queue_limit: queueLimit,
      queue_timeout_ms: this.queueTimeoutMs,
      session_queue_limit: this.sessionQueueLimit,
      active_sessions: this.activeSessions.size,
      channels,
    };
  }
}

module.exports = { ChannelConcurrencyManager };
