import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock the API client so the auth actions don't hit the network.
vi.mock('@/lib/api', () => ({
  authApi: {
    login: vi.fn(async () => ({
      access_token: 'tok',
      token_type: 'bearer',
      user: { id: 'u1', email: 'a@b.com' },
    })),
    register: vi.fn(async () => ({
      access_token: 'tok',
      token_type: 'bearer',
      user: { id: 'u1', email: 'a@b.com' },
    })),
    logout: vi.fn(async () => undefined),
    me: vi.fn(async () => ({ id: 'u1', email: 'a@b.com' })),
  },
}));

import { useAppStore } from '@/lib/store';
import { authApi } from '@/lib/api';

beforeEach(() => {
  useAppStore.setState({
    user: null,
    accessToken: null,
    isAuthenticated: false,
    isLoading: false,
    conversations: [],
    activeId: null,
    ragFiles: [],
  });
  vi.clearAllMocks();
});

describe('app store — auth', () => {
  it('login action sets authenticated state from the response', async () => {
    await useAppStore.getState().login('a@b.com', 'password');
    const s = useAppStore.getState();
    expect(s.isAuthenticated).toBe(true);
    expect(s.user?.id).toBe('u1');
    expect(s.user?.email).toBe('a@b.com');
    expect(s.accessToken).toBe('tok');
  });

  it('register action authenticates the new user', async () => {
    await useAppStore.getState().register('a@b.com', 'password');
    expect(useAppStore.getState().isAuthenticated).toBe(true);
    expect(authApi.register).toHaveBeenCalledOnce();
  });

  it('logout action clears auth state', async () => {
    await useAppStore.getState().login('a@b.com', 'password');
    await useAppStore.getState().logout();
    const s = useAppStore.getState();
    expect(s.isAuthenticated).toBe(false);
    expect(s.user).toBeNull();
    expect(s.accessToken).toBeNull();
    expect(authApi.logout).toHaveBeenCalledOnce();
  });

  it('fetchCurrentUser sets user when the session is valid', async () => {
    await useAppStore.getState().fetchCurrentUser();
    const s = useAppStore.getState();
    expect(s.isAuthenticated).toBe(true);
    expect(s.isLoading).toBe(false);
    expect(s.user?.id).toBe('u1');
  });

  it('fetchCurrentUser clears state when unauthenticated', async () => {
    (authApi.me as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('no session'));
    await useAppStore.getState().fetchCurrentUser();
    const s = useAppStore.getState();
    expect(s.isAuthenticated).toBe(false);
    expect(s.isLoading).toBe(false);
    expect(s.user).toBeNull();
  });
});
