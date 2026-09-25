/** Prefix a root-relative path with the configured site base (/litemind-ui). */
export function u(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}${path}`;
}
