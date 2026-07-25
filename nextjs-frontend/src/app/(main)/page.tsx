import { redirect } from 'next/navigation';

/**
 * Root page — immediately redirects to /chat.
 *
 * This is a server component so the redirect happens on the server
 * with no client-side JavaScript required.
 */
export default function RootPage(): never {
  redirect('/chat');
}
